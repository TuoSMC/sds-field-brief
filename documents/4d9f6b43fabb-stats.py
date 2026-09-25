#!/usr/bin/env python3
"""Statistics from the immutable SO ledger and checked-in reference snapshots.

Vendor views use multi-membership; whole-corpus views count each SO once. Amounts
are document totals, not recognized revenue, and must not be added across vendors.
PO-set deduplication is used only for distinct_deals, deal extremes, and DDN's split.
For extremes, the latest dated, valued SO represents a PO set; DDN uses all evidence
in the set. Missing software totals are derived only from fully priced SO lines.
MBD parts are excluded from model/series counts; series shares use the specified
denominator of all SOs with any machine token, including MBD-only SOs.

Salesperson concentration ranks SOs and value independently, within attributable
SOs. Tiers use interpolated value tertiles, with ties retained in the lower tier.
Value buckets are [0,50k), [50k,250k), [250k,1M], and (1M,infinity).
The command writes only vendor_stats.json; build_stats does not mutate its inputs.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parent
INDUSTRIES = (
    "AI/GPU 雲", "AI 新創/軟體", "媒體娛樂", "國家實驗室/HPC", "國防/航太",
    "大學/研究機構", "生醫/醫療", "金融/交易", "電信", "半導體/電子", "企業/其他",
    "政府/公用事業", "經銷商/終端未知", "未知",
)
VALUE_BUCKETS = ("<50k", "50k-250k", "250k-1M", ">1M")
MACHINE_PREFIXES = ("SYS-", "SSG-", "ASG-", "AS-", "CSE-", "SRS-")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def quantiles(values) -> dict:
    xs = sorted(v for v in values if number(v) is not None)
    return {"n": len(xs), "min": xs[0] if xs else None,
            "median": median(xs) if xs else None, "max": xs[-1] if xs else None}


def median_of(values):
    return quantiles(values)["median"]


def ranked_counts(values, limit=None) -> list:
    return sorted(Counter(values).items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def load_industry_rules(path: Path = BASE / "industry_map.csv") -> list:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["pattern", "canonical", "industry"]:
            raise ValueError("industry_map.csv must have pattern,canonical,industry columns")
        rules = []
        for row in reader:
            if not row["pattern"] or row["industry"] not in INDUSTRIES:
                raise ValueError(f"Invalid industry rule: {row}")
            rules.append((re.compile(row["pattern"], re.I), row["canonical"], row["industry"]))
        return rules


def classify(deal: dict, rules: list) -> tuple[str, str, str]:
    for basis in ("end_user", "folder_tag", "sold_to_raw"):
        text = " ".join((deal.get(basis) or "").split())
        if not text:
            continue
        if "ibm" in text.casefold():
            return "IBM", "半導體/電子", basis
        for pattern, canonical, industry in rules:
            if pattern.search(text):
                return canonical or text, industry, basis
    raw = next((deal.get(k) for k in ("end_user", "folder_tag", "sold_to_raw") if deal.get(k)), "未知")
    return " ".join(raw.split()), "未知", "none"


def memberships(deal: dict) -> list[str]:
    return sorted({v for v in deal.get("vendors", []) if v and v.casefold() != "tsmc"})


def quote_refs(deal: dict) -> set:
    return {str(q).strip() for key in ("quotes", "cq_quotes") for q in deal.get(key, []) if q}


def software_total(deal: dict):
    if "sw_total" in deal:
        return number(deal["sw_total"])
    lines = deal.get("sw_lines", [])
    if not lines or deal.get("sw_lines_unpriced") or any(number(s.get("ext")) is None for s in lines):
        return None
    return round(sum(s["ext"] for s in lines), 2)


def software_skus(deal: dict) -> list[str]:
    return sorted({line["sku"] for key in ("sw_lines", "sw_lines_unpriced")
                   for line in deal.get(key, []) if line.get("sku")})


def po_groups(deals: list) -> list[list]:
    groups = defaultdict(list)
    for d in deals:
        pos = frozenset(str(p) for p in d.get("po_numbers", []) if p)
        key = ("po", pos) if pos else ("so", d["so"])
        groups[key].append(d)
    return list(groups.values())


def representative(group: list) -> dict:
    return max(group, key=lambda d: (number(d.get("total_order")) is not None,
                                     d.get("so_date") or "", d["so"]))


def correlation(deals: list):
    pairs = [(d.get("so_to_po_days"), d.get("total_order")) for d in deals
             if number(d.get("so_to_po_days")) is not None and number(d.get("total_order")) is not None]
    if len(pairs) < 8:
        return None
    lags, values = zip(*pairs)
    if len(set(lags)) < 2 or len(set(values)) < 2:
        return {"spearman_rho": None, "p": None, "n": len(pairs)}
    rho, p = spearmanr(lags, values)
    return {"spearman_rho": float(rho), "p": float(p), "n": len(pairs)}


def value_bucket(value):
    if number(value) is None:
        return None
    return ("<50k" if value < 50_000 else "50k-250k" if value < 250_000
            else "250k-1M" if value <= 1_000_000 else ">1M")


def weka_ra_match(deals: list, reference: dict) -> dict:
    ra = reference["weka_ra"]
    allowed = {s.upper() for key in ("bundles_all", "pcie4", "certified") for s in ra[key]}
    stems = tuple("-".join(s.split("-")[:3]) for s in sorted(allowed))
    mapping = {k.upper(): v.upper() for k, v in ra["chassis_to_system"].items()}
    counts = dict.fromkeys(("in_ra", "outside_ra", "unknown_hw"), 0)
    for d in deals:
        tokens = {m.upper() for m in d.get("machines", [])}
        if not tokens:
            counts["unknown_hw"] += 1
            continue
        candidates = tokens | {mapping[t] for t in tokens if t in mapping}
        hit = any(t in allowed or t.startswith(stems) for t in candidates)
        counts["in_ra" if hit else "outside_ra"] += 1
    return counts


def ddn_product_split(groups: list, reference: dict) -> dict:
    appliance = tuple(p.upper() for p in reference["ddn"]["appliance_sku_prefixes"])
    infinia = tuple(p.upper() for p in reference["ddn"]["infinia_sku_prefixes"])
    counts = dict.fromkeys(("appliance", "infinia", "unknown"), 0)
    for group in groups:
        machines = {m.upper() for d in group for m in d.get("machines", [])}
        skus = {s.upper() for d in group for s in software_skus(d)}
        skus.update(str(line[k]).upper() for d in group for line in d.get("po_sw_lines", [])
                    for k in ("sku", "alias") if line.get(k))
        skus.update(s.upper() for d in group for s in d.get("bundle_sku", []))
        label = ("appliance" if any(m.startswith(appliance) for m in machines) else
                 "infinia" if any(s.startswith(infinia) for s in machines | skus) else "unknown")
        counts[label] += 1
    return counts


def pack_deal(deal: dict) -> dict:
    keys = ("so", "customer", "industry", "so_date", "total_order", "sw_total", "machines",
            "so_to_po_days", "software_only", "salesperson")
    return {**{k: deal.get(k) for k in keys}, "sw_skus": software_skus(deal)}


def vendor_stats(vendor: str, deals: list, reference: dict) -> dict:
    industries = dict(ranked_counts(d["industry"] for d in deals))
    by_industry = {i: [d for d in deals if d["industry"] == i] for i in industries}
    groups = po_groups(deals)
    distinct = [representative(g) for g in groups]
    valued = [d for d in distinct if number(d.get("total_order")) is not None]
    lags = [[d["so_to_po_days"], d.get("total_order"), d["customer"], d["so"]] for d in deals
            if number(d.get("so_to_po_days")) is not None]
    margins_by_sku = defaultdict(list)
    for d in deals:
        for m in d.get("sw_line_margins", []):
            cost, sell, margin = (number(m.get(k)) for k in ("cost_unit", "sell_unit", "margin_pct"))
            if cost is not None and sell is not None and margin is not None and cost >= 2 and sell >= 2 and -50 <= margin <= 60:
                margins_by_sku[m["sku"]].append(margin)
    shares = []
    for d in deals:
        sw, subtotal = d["sw_total"], number(d.get("subtotal"))
        if sw is not None and subtotal is not None and subtotal > 0 and sw <= subtotal * 1.001:
            shares.append(sw / subtotal)
    values = [d["total_order"] for d in deals if number(d.get("total_order")) is not None]
    result = {
        "so_count": len(deals), "distinct_deals": len(groups),
        "years": dict(sorted(Counter(d.get("year_folder") or "未知" for d in deals).items())),
        "industry_so": industries,
        "industry_value_usd": {
            i: {"sum": round(sum(d["total_order"] for d in xs if number(d.get("total_order")) is not None), 2),
                "n_with_value": sum(number(d.get("total_order")) is not None for d in xs)}
            for i, xs in by_industry.items()},
        "industry_hw_named": {
            i: {label: sum(("有寫機型" if d.get("machines") else "純軟體" if d.get("software_only") else "未知") == label for d in xs)
                for label in ("有寫機型", "純軟體", "未知")} for i, xs in by_industry.items()},
        "industry_year_so": {i: dict(sorted(Counter(d.get("year_folder") or "未知" for d in xs).items()))
                             for i, xs in by_industry.items()},
        "top_customers": ranked_counts((d["customer"] for d in deals), 12),
        "with_named_machine": sum(bool(d.get("machines")) for d in deals),
        "software_only_flagged": sum(bool(d.get("software_only")) for d in deals),
        "total_order": quantiles(values), "total_order_sum": round(sum(values), 2) if values else None,
        "sw_share_of_subtotal": quantiles(shares),
        "sw_line_margin_pct": quantiles(m for xs in margins_by_sku.values() for m in xs),
        "sw_line_margin_by_sku": {sku: quantiles(xs) for sku, xs in
                                  sorted(margins_by_sku.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:8]},
        "so_to_po_days": quantiles(row[0] for row in lags),
        "lag_by_industry": {i: {k: v for k, v in quantiles(d.get("so_to_po_days") for d in xs).items() if k != "min"}
                            for i, xs in by_industry.items()},
        "lag_vs_value_spearman": correlation(deals),
        "fastest": sorted(lags, key=lambda row: (row[0], row[3]))[:3],
        "slowest": sorted(lags, key=lambda row: (-row[0], row[3]))[:3],
        "slowest_with_value": sorted((row for row in lags if number(row[1]) is not None), key=lambda row: (-row[0], row[3]))[:3],
        "largest_deals": [pack_deal(d) for d in sorted(valued, key=lambda d: (-d["total_order"], d["so"]))[:5]],
        "smallest_deals": [pack_deal(d) for d in sorted(valued, key=lambda d: (d["total_order"], d["so"]))[:5]],
        "multi_quote_so": sum(len(quote_refs(d)) > 1 for d in deals),
        "with_quote_ref": sum(bool(quote_refs(d)) for d in deals),
    }
    if vendor == "Weka":
        result["ra_match"] = weka_ra_match(deals, reference)
    if vendor == "DDN":
        result["product_split"] = ddn_product_split(groups, reference)
    return result


def machine_stats(deals: list, series_map: dict) -> tuple[list, list, dict]:
    rules = [(re.compile(r["pattern"], re.I), r["series"]) for r in series_map["rules"]]
    model_sos, model_vendors, series_sos = defaultdict(set), defaultdict(set), defaultdict(set)
    vendor_series = defaultdict(lambda: defaultdict(set))
    model_series = {}
    for d in deals:
        for model in set(d.get("machines", [])):
            if not model.upper().startswith(MACHINE_PREFIXES):
                continue
            series = next((label for pattern, label in rules if pattern.search(model)), "unknown")
            model_series[model] = series
            model_sos[model].add(d["so"])
            model_vendors[model].update(d["vendors"])
            series_sos[series].add(d["so"])
            for vendor in d["vendors"]:
                vendor_series[vendor][series].add(d["so"])
    models = [{"name": m, "so_count": len(sos), "vendors": sorted(model_vendors[m]), "series": model_series[m]}
              for m, sos in model_sos.items()]
    models.sort(key=lambda row: (-row["so_count"], row["name"]))
    denominator = sum(bool(d.get("machines")) for d in deals)
    series = [{"series": label, "so_count": len(sos), "share": len(sos) / denominator if denominator else None,
               "top_model": next(m["name"] for m in models if m["series"] == label)}
              for label, sos in series_sos.items()]
    series.sort(key=lambda row: (-row["so_count"], row["series"]))
    vendors = {v: {s: len(sos) for s, sos in sorted(vendor_series[v].items(), key=lambda kv: (-len(kv[1]), kv[0]))}
               for v in sorted({v for d in deals for v in d["vendors"]})}
    return models, series, vendors


def salesperson_stats(deals: list, by_vendor: dict) -> tuple[list, dict, dict]:
    by_person = defaultdict(list)
    for d in deals:
        if d.get("salesperson"):
            by_person[d["salesperson"]].append(d)
    people = []
    for name, xs in by_person.items():
        values = [d["total_order"] for d in xs if number(d.get("total_order")) is not None]
        kinds = {d.get("salesperson_kind", "unknown") for d in xs}
        people.append({
            "name": name, "kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
            "so_count": len(xs), "usd_sum": round(sum(values), 2), "n_with_value": len(values),
            "vendors": sorted({v for d in xs for v in d["vendors"]}),
            "industries": sorted({d["industry"] for d in xs}),
            "median_lag": median_of(d.get("so_to_po_days") for d in xs),
            "software_only_so": sum(bool(d.get("software_only")) for d in xs),
            "reissue_so": sum(bool(d.get("reissue_flags")) for d in xs),
            "cq_mean": mean(d.get("cq_count", 0) for d in xs),
            "quotes_per_so": mean(len(quote_refs(d)) for d in xs), "tier": "未知",
        })
    values = [p["usd_sum"] for p in people if p["n_with_value"]]
    if values:
        low, high = np.quantile(values, [1 / 3, 2 / 3])
        for p in people:
            if p["n_with_value"]:
                p["tier"] = "小" if p["usd_sum"] <= low else "中" if p["usd_sum"] <= high else "大"
    people.sort(key=lambda row: (-row["so_count"], row["name"]))
    total_sos, total_usd = sum(p["so_count"] for p in people), sum(p["usd_sum"] for p in people)
    concentration = {
        "n_people": len(people),
        "top5_share_so": sum(p["so_count"] for p in people[:5]) / total_sos if total_sos else None,
        "top5_share_usd": sum(sorted((p["usd_sum"] for p in people), reverse=True)[:5]) / total_usd if total_usd else None,
    }
    vendor_people = {v: ranked_counts((d["salesperson"] for d in xs if d.get("salesperson")), 8)
                     for v, xs in by_vendor.items()}
    return people, concentration, vendor_people


def friction_summary(deals: list, per_so: dict) -> dict:
    return {
        "n": len(deals), "median_lag": median_of(d.get("so_to_po_days") for d in deals),
        "reissue_share": sum(bool(d.get("reissue_flags")) for d in deals) / len(deals) if deals else None,
        "cq_per_so_median": median_of(d.get("cq_count", 0) for d in deals),
        "email_span_median": median_of(per_so.get(d["so"], {}).get("span_days") for d in deals),
    }


def build_stats(ledger: dict, emails: dict, combo: dict, reference: dict,
                series_map: dict, industry_rules: list) -> dict:
    deals = []
    for raw in ledger["deals"]:
        customer, industry, basis = classify(raw, industry_rules)
        deals.append({**raw, "vendors": memberships(raw), "customer": customer, "industry": industry,
                      "industry_basis": basis, "sw_total": software_total(raw)})
    if len({d["so"] for d in deals}) != len(deals):
        raise ValueError("The ledger must have one record per SO")
    by_vendor = {v: [d for d in deals if v in d["vendors"]]
                 for v in sorted({v for d in deals for v in d["vendors"]})}
    actual = {v: len(xs) for v, xs in by_vendor.items()}
    expected = {v: row["so_count"] for v, row in combo["per_vendor"].items()}
    if actual != expected:
        raise ValueError(f"Ledger/combo vendor membership mismatch: {actual} != {expected}")
    vendors = {v: vendor_stats(v, xs, reference) for v, xs in
               sorted(by_vendor.items(), key=lambda kv: (-len(kv[1]), kv[0]))}
    machines, series, vendor_series = machine_stats(deals, series_map)
    people, concentration, vendor_people = salesperson_stats(deals, by_vendor)
    tiers = {p["name"]: p["tier"] for p in people}
    per_so = emails["per_so"]
    reissues = sum(bool(d.get("reissue_flags")) for d in deals)
    friction = {f"{k}_files": emails["keyword_file_counts"].get(k, 0)
                for k in ("end_customer_info", "cq", "bq", "ibom", "approval", "review")}
    friction.update({
        "reissue_so": reissues, "reissue_share": reissues / len(deals) if deals else None,
        "bundle_sku_so": sum(bool(d.get("bundle_sku")) for d in deals),
        "email_span_days": quantiles(per_so.get(d["so"], {}).get("span_days") for d in deals),
        "distinct_senders": quantiles(per_so.get(d["so"], {}).get("distinct_senders") for d in deals),
        "cq_per_so": quantiles(d.get("cq_count", 0) for d in deals),
    })
    bucket_deals = {b: [d for d in deals if value_bucket(d.get("total_order")) == b] for b in VALUE_BUCKETS}
    roster = {r["vendor_key"] for r in reference["roster"]}
    weekly, cabinets = set(reference["weekly_report_rows"]), set(reference["cabinets"])
    customers = defaultdict(set)
    for d in deals:
        if d["industry"] not in ("經銷商/終端未知", "未知"):
            customers[d["customer"]].update(d["vendors"])
    refs = {q for d in deals for q in quote_refs(d)}
    if "distinct_quote_numbers" in ledger and len(refs) != ledger["distinct_quote_numbers"]:
        raise ValueError("Ledger quote-reference count differs from the observed quote union")
    return {
        "vendors": vendors, "machines_all": machines, "series_all": series, "series_by_vendor": vendor_series,
        "salespersons": people, "salesperson_concentration": concentration, "salesperson_by_vendor": vendor_people,
        "friction": friction,
        "friction_by_value_bucket": {b: friction_summary(xs, per_so) for b, xs in bucket_deals.items()},
        "friction_by_salesperson_tier": {t: friction_summary([d for d in deals if tiers.get(d.get("salesperson")) == t], per_so)
                                         for t in ("大", "中", "小")},
        "program_matrix": [{"vendor": v, "on_sds_page": v in roster, "weekly_row": v in weekly,
                            "cabinet_so_count": combo["per_vendor"].get(v, {}).get("so_count", 0)}
                           for v in sorted(roster | weekly | cabinets)],
        "customers_with_multiple_vendors": {c: sorted(vs) for c, vs in sorted(customers.items()) if len(vs) > 1},
        "all_vendors_lag_vs_value": correlation(deals),
        "lag_by_value_bucket": {b: quantiles(d.get("so_to_po_days") for d in xs) for b, xs in bucket_deals.items()},
        "industry_all": dict(ranked_counts(d["industry"] for d in deals)),
        "distinct_quote_refs": len(refs), "lag_excluded_n": sum(bool(d.get("lag_excluded")) for d in deals),
        "deals_classified": [{k: d.get(k) for k in ("so", "vendors", "customer", "industry", "industry_basis",
                                                    "total_order", "so_to_po_days", "salesperson")} for d in deals],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    report = build_stats(read_json(BASE / "deal_ledger.json"), read_json(BASE / "email_evidence.json"),
                         read_json(BASE / "combo_results.json"), read_json(BASE / "reference/supermicro_sds_2026-09-22.json"),
                         read_json(BASE / "reference/series_map.json"), load_industry_rules())
    (BASE / "vendor_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote vendor_stats.json: {len(report['deals_classified'])} SOs, {len(report['vendors'])} vendors, {report['distinct_quote_refs']} quote refs")


if __name__ == "__main__":
    main()
