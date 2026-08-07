# Tenant value roadmap — what a shop inside the mall gets

**Framing (non-negotiable).** Everyone sells footfall; it is a commodity and we
lose that race. Our edge is the layer above it: not *how many walked in*, but
**how many were interested, how many hesitated, and where we lost them**. Every
metric below must answer a question a footfall counter cannot. If a feature only
produces "N people entered", it belongs to someone else's product.

The mall operator and the shop want different things, and both are payers:

| | Shop (tenant) | Mall operator (landlord) |
|---|---|---|
| Question | "Why didn't they buy?" | "What is this unit worth?" |
| Metric | capture, hesitation, abandonment | corridor traffic → tenant conversion |
| Sells | merchandising decisions | rent justification, unit pricing |

---

## Phase A — The storefront funnel *(mostly existing parts, wired together)*

The single most valuable thing we can give a shop, because it is the first place
revenue leaks. Each stage already has an engine primitive; what is missing is the
**visit** object that links them.

```
passed by  →  looked at window  →  entered  →  stayed  →  engaged  →  reached
   ↑ traffic     ↑ attention        ↑ line      ↑ visit    ↑ zone     ↑ reach
```

**To build**
1. **Visit objects.** Pair `enter` and `exit` crossings per identity into a visit
   with a duration. Today we only count crossings. This unlocks every dwell
   metric below and is the foundation of the phase.
2. **Dwell distribution**, not just an average: p50/p90 and a histogram. An
   average of 4 minutes hides "half leave in 20 seconds".
3. **Bounce rate** — entered and left under a threshold (default 30 s) without
   engaging any surface. This is the "walked in, couldn't decide, left" number
   the shop actually feels but cannot measure.
4. **Capture rate** — already shipped (`enters ÷ passers-by`).

**Honest limits to state in the report:** without re-identification we cannot
follow a person between cameras, so a visit is per entrance line. Someone
entering through one door and leaving by another counts as an unmatched visit —
reported as such, never silently guessed.

---

## Phase B — Decision signals *(our actual differentiator)*

Footfall products stop at Phase A. This is where the attention engine earns its
price, and it is why we can charge a shop instead of a facilities manager.

- **Window conversion** — SHIPPED. Of the people who *looked* at the window, how
  many came in? Low looks = wrong display. High looks but low entry = the window
  promises something the shop doesn't deliver.
- **Hesitation** — SHIPPED. Long dwell (≥3 s), repeated glances (≥2), **no
  reach**. The clearest "interested but not convinced" signal we can produce, and
  it maps directly to a merchandising or pricing action.
- **Abandonment point** — SHIPPED. The last surface engaged before the exit,
  reported per surface. Tells the shop *where* it lost the customer, not just
  that it did.
- **Reach-to-dwell ratio** — DROPPED as a separate metric. `reach_rate` (share of
  lookers who reached) already separates a display people admire from one they
  touch, and `hesitation` covers the "looked long, never reached" case. A third
  ratio over the same two numbers would add a figure, not information.

---

## Phase C — Recommendations *(turn measurement into an instruction)*

Numbers do not change behaviour; instructions do. Every recommendation must cite
the measurement and the comparison that produced it, or it is horoscope.

- **Self-benchmark** — this week vs. the shop's own trailing weeks. Always valid,
  needs no other tenant's data.
- **Category benchmark** — SHIPPED (first slice). Each surface now carries its
  percentile against the same surface type across the anonymous dataset
  (`benchmark_percentile`), withheld below 20 comparable episodes. Never against
  a named neighbour.
- **Rules with evidence**, e.g. "Capture rate 11% vs. your 4-week median 19% —
  the drop starts Tuesday, when the window display changed." Withhold the
  recommendation when the sample is too small to support it, exactly as we
  withhold implausible surface sizes today.
- **Time-of-day view** — SHIPPED. Hourly capture/attention profile over the last
  N days with the weakest hour named (`/api/cameras/{id}/hourly`, shown in the
  live report). Hours with fewer than 20 people show no rate: a "0% capture"
  from three passers-by would send someone chasing a problem that isn't there.

---

## Phase D — Mall operator view

- Corridor traffic → per-tenant capture: which units convert the footfall the
  mall delivers, and which waste it.
- Unit value map: attention and capture by location, the evidence base for rent
  and for pricing common-area advertising.
- Tenant scorecard the mall can hand over at renewal.

---

## Sequencing and cost

| Phase | Depends on | Rough effort | Sellable on its own |
|---|---|---|---|
| A | visit pairing | days | Yes — this is the pilot deliverable |
| B | A + existing zones/reach | days | Yes — this is the reason to renew |
| C | B + several weeks of data | weeks | Only after data accumulates |
| D | multiple cameras/tenants | weeks | Sells to the landlord, not the shop |

**Do A and B for the pilot.** C needs history that does not exist yet on day one
— promising it before the data exists would be the same mistake as reporting a
surface size the depth map cannot support. D follows once more than one unit is
instrumented.

## What this roadmap deliberately excludes

Queue management, staff scheduling, occupancy/safety limits, POS conversion.
They are the RetailNext/Sensormatic ring: commodity, hardware-partnered, priced
down. Each would pull the product toward operations analytics and away from the
measurement position we are building.

---

## On public datasets (what they can and cannot settle)

Public sets are useful for tuning the engine, not for the accuracy claim.

| Question | Public data can answer it? |
|---|---|
| Do we detect and count people correctly? | **Yes** — MOT-format pedestrian sets (`tools/eval_mot.py`). This settles the traffic denominator, our largest open uncertainty. |
| Do we track identities stably? | Partly — same sets, ID metrics. |
| Do we detect a reach at a shelf? | Partly — shopper-action sets label reach/retract. |
| **Is "looked at that surface" correct?** | **No.** No public pedestrian dataset labels gaze against a declared surface, and that is the metric we sell. |

Two consequences we hold to:

1. The published accuracy figure must come from footage we have permission to
   use. Research sets are almost always research-only; a company selling audit
   trust cannot base its evidence file on a licence it is breaching.
2. Even with a permissive licence, the core claim would still be unproven,
   because the label we need does not exist in them. `tools/labeler.html` stays
   on the critical path.
