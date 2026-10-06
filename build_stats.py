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

import pyarrow.compute as pc
import pyarrow.parquet as pq

TEAMS = ([f"t1_b{i}_name" for i in range(3)], [f"t2_b{i}_name" for i in range(3)])
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


def main(path, ranks=None, out=STATS_DIR / "gizemli.json", weights=None, tr=None):
    t = pq.read_table(path, columns=["map", "record", "avg_elo", "battle_time", *TEAMS[0], *TEAMS[1]])
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
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(stats, separators=(",", ":")), encoding="utf-8")
    print(f"{t.num_rows} set, {len(stats['all'])} savaşçı, {len(stats['map'])} harita -> {out}")


if __name__ == "__main__":
    tr = tr_names(set(pq.read_table(sys.argv[1], columns=["map"])["map"].unique().drop_null().to_pylist()))
    for name, ranks in BANDS.items():
        main(sys.argv[1], ranks, STATS_DIR / f"{name}.json", WEIGHTS.get(name), tr)
