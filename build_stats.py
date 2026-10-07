"""Hugging Face ranked veri setinden stats.json üretir.

stats/<rank>.json hem engine.py'nin (varsayılan gizemli) hem mobil uygulamanın veri dosyası; uygulamada rank seçilir.
Veri: https://huggingface.co/datasets/EliF77/brawlstars-ranked (MIT), sezon başına bir parquet.
Kullanım: python build_stats.py data/season54.parquet   (BANDS'deki her rank için stats/<rank>.json)
GitHub Actions (.github/workflows/stats.yml) bunu her gün en yeni sezonla çalıştırıp stats/'ı depoya koyar; uygulama oradan indirir.
Rank = API'nin soloRanked basamağı (1 Bronz I ... 10 Elmas I, 13 Mitik I, 16 Efsanevi I, 19 Usta I, 22 Pro);
aralık verilirse yalnız altı oyuncunun ortalaması bu aralıkta olan setler sayılır.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import optimize, sparse

TEAMS = ([f"t1_b{i}_name" for i in range(3)], [f"t2_b{i}_name" for i in range(3)])
ELO = ([f"t1_b{i}_elo" for i in range(3)], [f"t2_b{i}_elo" for i in range(3)])
LAM = 100  # model.py: season54'te üç bantta da en iyi L2 cezası (30-3000 tarandı)
# Usta (19+) season54'te tek başına ~4K set: Efsanevi'ye katıldı.
BANDS = {"elmas": "10-12", "gizemli": "13-15", "efsanevi": "16-22"}
# Banda özel ağırlık (stats json "weights"; eksik anahtar engine.WEIGHTS'ten). Şu an boş: backtest .15'i her bantta en iyi buldu.
# Efsanevi pop .2 denendi (2026-10-05): Usta/Pro YouTube ilk 3 örtüşmesi %33 -> %38 ama backtest kazanma farkı artmadı -> geri alındı.
WEIGHTS = {}
STATS_DIR = Path(__file__).parent / "stats"


def acc(table, keys, win, sink):
    """table'ı keys'e göre gruplayıp (oyun, galibiyet) toplamlarını sink'e ekler."""
    for r in table.group_by(keys).aggregate([("g", "sum"), (win, "sum")]).to_pylist():
        k = tuple(r[c] for c in keys)
        g, w = sink.get(k, (0, 0))
        sink[k] = (g + r["g_sum"], w + r[f"{win}_sum"])


def nest(flat):
    """{(a, b): (g, w)} -> {a: {b: [g, w]}}"""
    out = {}
    for (a, b), gw in flat.items():
        out.setdefault(a, {})[b] = list(gw)
    return out


class Model:
    def __init__(self, names, maps):
        self.names, self.maps = names, maps
        self.ix = {b: i for i, b in enumerate(names)}
        self.mx = {m: i for i, m in enumerate(maps)}
        B, M = len(names), len(maps)
        self.B, self.M = B, M
        self.o_map, self.o_syn, self.o_vs = B, B + M * B, B + M * B + B * B
        self.o_elo = self.o_vs + B * B
        self.D = self.o_elo + 2  # elo, sabit

    def design(self, t):
        """-> X (sparse), w1, w2 (oyun sayıları); bilinmeyen savaşçı/haritalı setler atılır."""
        tm = [np.array([[self.ix.get(b, -1) for b in t[c].to_pylist()] for c in cols]).T for cols in TEAMS]
        m = np.array([self.mx.get(x, -1) for x in t["map"].to_pylist()])
        keep = (tm[0] >= 0).all(1) & (tm[1] >= 0).all(1) & (m >= 0)
        a, b, m = tm[0][keep], tm[1][keep], m[keep]
        n, B = len(m), self.B
        rows = np.arange(n)
        I, J, V = [], [], []

        def add(cols, vals):
            I.append(rows), J.append(cols), V.append(np.broadcast_to(vals, n).astype(np.float32))

        for team, s in ((a, 1), (b, -1)):
            for k in range(3):
                add(team[:, k], s)
                add(self.o_map + m * B + team[:, k], s)
            for i, j in ((0, 1), (0, 2), (1, 2)):
                x, y = team[:, i], team[:, j]
                add(self.o_syn + np.minimum(x, y) * B + np.maximum(x, y), s)
        for i in range(3):
            for j in range(3):
                x, y = a[:, i], b[:, j]
                add(self.o_vs + np.minimum(x, y) * B + np.maximum(x, y), np.where(x < y, 1, np.where(x > y, -1, 0)))
        elo = lambda cols: sum(t[c].to_numpy().astype(np.float32) for c in cols)[keep]
        add(np.full(n, self.o_elo), (elo(ELO[0]) - elo(ELO[1])) / 3)
        add(np.full(n, self.o_elo + 1), 1)
        X = sparse.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, self.D))
        rec = t["record"].to_pylist()
        w1 = np.array([r.count("T1") for r in rec], np.float64)[keep]
        w2 = np.array([r.count("T2") for r in rec], np.float64)[keep]
        return X, w1, w2

    def fit(self, X, w1, w2, lam):
        reg = np.ones(self.D)
        reg[self.o_elo:] = 0  # elo ve sabit cezasız
        XT = X.T.tocsr()

        def f(th):
            z = X @ th
            loss = (w1 * np.logaddexp(0, -z) + w2 * np.logaddexp(0, z)).sum() + lam / 2 * (reg * th * th).sum()
            g = XT @ ((w1 + w2) / (1 + np.exp(-z)) - w1) + lam * reg * th
            return loss, g

        self.th = optimize.minimize(f, np.zeros(self.D), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        B, M, th = self.B, self.M, self.th
        self.main = th[:B]
        self.mapb = th[self.o_map:self.o_syn].reshape(M, B)
        U = th[self.o_syn:self.o_vs].reshape(B, B)
        self.S = np.triu(U, 1) + np.triu(U, 1).T  # takım içi uyum
        U = np.triu(th[self.o_vs:self.o_elo].reshape(B, B), 1)
        self.V = U - U.T  # V[c, e]: c'nin e'ye karşı etkisi (logit)
        return self

    def logloss(self, X, w1, w2):
        z = X @ self.th
        p = 1 / (1 + np.exp(-z))
        g = w1 + w2
        ll = (w1 * np.logaddexp(0, -z) + w2 * np.logaddexp(0, z)).sum() / g.sum()
        brier = (w1 * (1 - p) ** 2 + w2 * p ** 2).sum() / g.sum()
        return ll, brier, p

    def export(self):
        """stats json "model": logit katsayıları ad adına; syn/vs yalnız ad sırasında a < b (syn simetrik, vs ters işaretli)."""
        r = lambda x: round(float(x), 4)
        n = self.names
        tri = lambda M: {a: {b: r(M[i, j]) for j, b in enumerate(n) if j > i and abs(M[i, j]) >= 1e-4} for i, a in enumerate(n)}
        return {"main": {b: r(x) for b, x in zip(n, self.main)},
                "map": {m: {b: r(x) for b, x in zip(n, row)} for m, row in zip(self.maps, self.mapb)},
                "syn": tri(self.S), "vs": tri(self.V)}


def tr_names(maps):
    """{Türkçe harita adı: İngilizce}: oyun Türkçeyken ekranda Türkçe ad çıkıyor. BrawlAPI her oyun güncellemesinde oyun
    dosyalarından (locations + localization) yeniden üretiyor. Alınamazsa boş; main eski dosyadakini korur."""
    try:
        # Python'un varsayılan User-Agent'ına 403 dönüyor.
        get = lambda p: json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://api.brawlapi.com/game/{p}", headers={"User-Agent": "genes-drafter"}), timeout=120))
        en, tr, loc = get("localization/texts"), get("localization/tr"), get("csv_logic/locations")
    except Exception as e:
        print("Türkçe harita adları alınamadı:", e)
        return {}
    tids = {r["TID"] for r in loc.values() if r.get("TID") in en and r["TID"] in tr}
    return {tr[t]["TR"]: en[t]["EN"] for t in tids if en[t]["EN"] in maps}


def main(path, ranks=None, out=STATS_DIR / "gizemli.json", weights=None, tr=None, fit=True):
    """fit: kazanma modelini de eğit (stats json "model"; backtest/model.py kendi eğitir, kapatır)."""
    t = pq.read_table(path, columns=["map", "record", "avg_elo", "battle_time", *TEAMS[0], *TEAMS[1], *ELO[0], *ELO[1]])
    valid = pc.is_valid(t["map"])
    if ranks:
        lo, hi = map(int, ranks.split("-"))
        valid = pc.and_(valid, pc.and_(pc.greater_equal(t["avg_elo"], lo), pc.less(t["avg_elo"], hi + 1)))
    for c in TEAMS[0] + TEAMS[1]:
        valid = pc.and_(valid, pc.is_valid(t[c]))
    t = t.filter(valid)
    # record "T2-T1-T1" gibi: set'teki her oyun ayrı sayılır
    w1 = pc.count_substring(t["record"], "T1")
    w2 = pc.count_substring(t["record"], "T2")
    t = t.append_column("w1", w1).append_column("w2", w2).append_column("g", pc.add(w1, w2))

    all_, map_, vs, with_ = {}, {}, {}, {}
    for team, other, win in ((TEAMS[0], TEAMS[1], "w1"), (TEAMS[1], TEAMS[0], "w2")):
        for b in team:
            acc(t, [b], win, all_)
            acc(t, ["map", b], win, map_)
            for e in other:
                acc(t, [b, e], win, vs)
            for a in team:
                if a != b:
                    acc(t, [b, a], win, with_)

    stats = {
        "source": Path(path).name + (f" rank {ranks}" if ranks else ""),
        "sets": t.num_rows,
        "updated": (pc.max(t["battle_time"]).as_py() or "")[:8],  # son maçın günü, YYYYMMDD (uygulama ana ekranda gösterir)
        "all": {b: list(gw) for (b,), gw in all_.items()},
        "map": nest(map_),
        "vs": nest(vs),
        "with": nest(with_),
        # Türkçe adlar alınamadıysa eski dosyadakiler kalır.
        "tr": tr or (json.loads(Path(out).read_text(encoding="utf-8")).get("tr", {}) if Path(out).exists() else {}),
        **({"weights": weights} if weights else {}),  # yoksa motorlar varsayılan ağırlıkları kullanır
    }
    if fit:
        # Tam kompozisyon kazanma modeli (model.py ile denendi, 2026-10-07): uygulama varsa bunu, yoksa eski puanı kullanır.
        ok = pc.is_valid(t[ELO[0][0]])
        for c in ELO[0][1:] + ELO[1]:
            ok = pc.and_(ok, pc.is_valid(t[c]))
        mdl = Model(sorted(stats["all"]), sorted(stats["map"]))
        stats["model"] = mdl.fit(*mdl.design(t.filter(ok)), LAM).export()
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(stats, separators=(",", ":")), encoding="utf-8")
    print(f"{t.num_rows} set, {len(stats['all'])} savaşçı, {len(stats['map'])} harita -> {out}")


if __name__ == "__main__":
    tr = tr_names(set(pq.read_table(sys.argv[1], columns=["map"])["map"].unique().drop_null().to_pylist()))
    for name, ranks in BANDS.items():
        main(sys.argv[1], ranks, STATS_DIR / f"{name}.json", WEIGHTS.get(name), tr)
