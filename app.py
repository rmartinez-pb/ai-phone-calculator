import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Voice AI Pricing - Simplified", layout="wide")
st.title("Voice AI Pricing - Simplified Comparison")

# -----------------------------
# Fixed provider assumptions
# -----------------------------
PROVIDERS = {
    # Pipecat
    "Pipecat (profile from calculator)": {
        "fixed_monthly": 0.0,
        "included_minutes": 0.0,
        "rate_per_min": 121.93 / 1500.0,
    },

    # LiveKit plans (component-accurate from your screenshots)
    # Build/Ship total: 0.2049/min = agent(0.01) + telephony(0.004) + llm(0.0004) + stt(0.0105) + tts(0.1800)
    # Scale total:      0.0969/min = agent(0.01) + telephony(0.004) + llm(0.0004) + stt(0.0105) + tts(0.0720)
    "LiveKit (Build profile)": {
        "type": "livekit",
        "fixed_monthly": 0.0,
        "included_agent_minutes": 1000.0,
        "included_telephony_minutes": 50.0,
        "inference_credits": 2.50,
        "rates": {
            "agent": 0.0100,
            "telephony": 0.0040,
            "llm": 0.0004,
            "stt": 0.0105,
            "tts": 0.1800,
        },
    },
    "LiveKit (Ship profile)": {
        "type": "livekit",
        "fixed_monthly": 50.0,
        "included_agent_minutes": 5000.0,
        "included_telephony_minutes": 100.0,
        "inference_credits": 5.00,
        "rates": {
            "agent": 0.0100,
            "telephony": 0.0040,
            "llm": 0.0004,
            "stt": 0.0105,
            "tts": 0.1800,
        },
    },
    "LiveKit (Scale profile)": {
        "type": "livekit",
        "fixed_monthly": 500.0,
        "included_agent_minutes": 50000.0,
        "included_telephony_minutes": 1000.0,
        "inference_credits": 50.00,
        "rates": {
            "agent": 0.0100,
            "telephony": 0.0040,
            "llm": 0.0004,
            "stt": 0.0105,
            "tts": 0.0720,
        },
    },

    # Neto
    "Neto (1 agent)": {
        "fixed_monthly": 1500.0,
        "included_minutes": 0.0,
        "rate_per_min": 0.20,
    },

    # Hona AI
    "Hona AI": {
        "fixed_monthly": 0.0,
        "included_minutes": 0.0,
        "rate_per_min": 0.95,
    },
}

NETO_ONBOARDING_ONE_TIME = 5000.0

# -----------------------------
# UI inputs (ONLY two)
# -----------------------------
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    sessions = st.number_input("Sessions per month", min_value=0, value=500, step=50)
with c2:
    minutes_per_session = st.number_input("Minutes per session", min_value=0.0, value=3.0, step=0.5)

total_minutes = float(sessions) * float(minutes_per_session)

with c3:
    st.metric("Total minutes / month", f"{total_minutes:,.0f}")

# Neto onboarding is ONE-TIME: show it, don't amortize into monthly
include_neto_onboarding = st.checkbox("Include Neto one-time onboarding ($5,000) in totals (shown separately)", value=False)

# Providers effective (no amortization applied)
providers_effective = {name: dict(p) for name, p in PROVIDERS.items()}

# -----------------------------
# Pricing helpers
# -----------------------------
def livekit_inference_rate(p: dict) -> float:
    r = p["rates"]
    return float(r["llm"]) + float(r["stt"]) + float(r["tts"])

def monthly_cost(minutes: float, p: dict) -> float:
    """Monthly recurring cost only (no one-time onboarding baked in)."""
    if p.get("type") != "livekit":
        included = float(p.get("included_minutes", 0.0))
        over = max(0.0, minutes - included)
        return float(p["fixed_monthly"]) + over * float(p["rate_per_min"])

    r = p["rates"]
    fixed = float(p["fixed_monthly"])

    inc_agent = float(p["included_agent_minutes"])
    inc_tel = float(p["included_telephony_minutes"])
    credits = float(p["inference_credits"])

    agent_over = max(0.0, minutes - inc_agent)
    tel_over = max(0.0, minutes - inc_tel)

    agent_cost = agent_over * float(r["agent"])
    telephony_cost = tel_over * float(r["telephony"])

    inf_rate = livekit_inference_rate(p)
    inf_gross = minutes * inf_rate
    inf_net = max(0.0, inf_gross - credits)

    return fixed + agent_cost + telephony_cost + inf_net

def total_cost_including_one_time(minutes: float, name: str, p: dict) -> float:
    """Total used for ranking/plots if the user wants to include Neto onboarding as a one-time add-on."""
    base = monthly_cost(minutes, p)
    if include_neto_onboarding and name.startswith("Neto"):
        return base + NETO_ONBOARDING_ONE_TIME
    return base

def display_rate(p: dict) -> float:
    if p.get("type") == "livekit":
        r = p["rates"]
        return float(r["agent"]) + float(r["telephony"]) + livekit_inference_rate(p)
    return float(p.get("rate_per_min", 0.0))

def display_included(p: dict) -> float:
    if p.get("type") == "livekit":
        return float(p.get("included_agent_minutes", 0.0))
    return float(p.get("included_minutes", 0.0))

# -----------------------------
# Scenario totals table (single table, explicit components)
# -----------------------------
M = float(total_minutes)

rows = []
for name, p in providers_effective.items():
    monthly = monthly_cost(M, p)
    total = total_cost_including_one_time(M, name, p)    
    rows.append({
        "Provider": name,
        "One-time onboarding ($)": NETO_ONBOARDING_ONE_TIME if (include_neto_onboarding and name.startswith("Neto")) else 0.0,
        "Fixed ($/mo)": float(p["fixed_monthly"]),
        "Included min (display)": display_included(p),
        "Effective $/min (display)": display_rate(p),
        "Total ($/mo)": total
    })



# overdetailed table with explicit calculations (not used in final version, but kept here for reference and potential future use if we want to show explicit breakdowns in a table format instead of just the waterfall)
# rows = []
# for name, p in providers_effective.items():
#     monthly = monthly_cost(M, p)
#     total = total_cost_including_one_time(M, name, p)

#     if p.get("type") != "livekit":
#         fixed = float(p["fixed_monthly"])
#         included = float(p.get("included_minutes", 0.0))
#         rate = float(p.get("rate_per_min", 0.0))
#         over = max(0.0, M - included)
#         usage = over * rate
#         one_time = NETO_ONBOARDING_ONE_TIME if (include_neto_onboarding and name.startswith("Neto")) else 0.0

#         rows.append({
#             "Provider": name,
#             "Fixed ($/mo)": fixed,
#             "Included min": included,
#             "Overage min": over,
#             "Rate ($/min)": rate,
#             "Usage ($/mo)": usage,
#             "Monthly recurring ($)": monthly,
#             "One-time onboarding ($)": one_time,
#             "Total incl one-time ($)": total,
#         })
#     else:
#         r = p["rates"]
#         fixed = float(p["fixed_monthly"])
#         inc_agent = float(p["included_agent_minutes"])
#         inc_tel = float(p["included_telephony_minutes"])
#         credits = float(p["inference_credits"])

#         agent_over = max(0.0, M - inc_agent)
#         tel_over = max(0.0, M - inc_tel)

#         agent_rate = float(r["agent"])
#         tel_rate = float(r["telephony"])
#         llm_rate = float(r["llm"])
#         stt_rate = float(r["stt"])
#         tts_rate = float(r["tts"])
#         inf_rate = llm_rate + stt_rate + tts_rate

#         agent_cost = agent_over * agent_rate
#         tel_cost = tel_over * tel_rate

#         inf_gross = M * inf_rate
#         credit_used = min(credits, inf_gross)
#         inf_net = max(0.0, inf_gross - credits)

#         rows.append({
#             "Provider": name,
#             "Plan ($/mo)": fixed,
#             "Included agent min": inc_agent,
#             "Agent over min": agent_over,
#             "Agent $/min": agent_rate,
#             "Agent ($/mo)": agent_cost,
#             "Included tel min": inc_tel,
#             "Tel over min": tel_over,
#             "Tel $/min": tel_rate,
#             "Telephony ($/mo)": tel_cost,
#             "LLM $/min": llm_rate,
#             "STT $/min": stt_rate,
#             "TTS $/min": tts_rate,
#             "Inference $/min": inf_rate,
#             "Inference gross ($/mo)": inf_gross,
#             "Credits used ($/mo)": credit_used,
#             "Inference net ($/mo)": inf_net,
#             "Monthly recurring ($)": monthly,
#             "One-time onboarding ($)": 0.0,
#             "Total incl one-time ($)": total,
#         })

df_point = pd.DataFrame(rows)
st.subheader("Totals at your scenario")
st.dataframe(df_point, use_container_width=True)

# -----------------------------
# Consolidated curves plot (Plotly)
# -----------------------------
st.subheader("Total monthly cost curves (all providers)")

max_minutes_default = max(1000.0, M * 2.0)
max_minutes_default = float(int(max_minutes_default // 1000 * 1000))
max_minutes = st.number_input("Max minutes shown on curve", min_value=1000.0, value=max_minutes_default, step=1000.0)
step = st.number_input("Step (minutes)", min_value=100.0, value=500.0, step=100.0)

minutes_grid = np.arange(0, max_minutes + step, step)

fig = go.Figure()
for name, p in providers_effective.items():
    y = [total_cost_including_one_time(m, name, p) for m in minutes_grid]

    fig.add_trace(go.Scatter(
        x=minutes_grid,
        y=y,
        mode="lines",
        name=name
    ))

    scenario_cost = total_cost_including_one_time(M, name, p)
    fig.add_trace(go.Scatter(
        x=[M],
        y=[scenario_cost],
        mode="markers",
        name=f"{name} (scenario)",
        marker=dict(size=10),
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Minutes: {M:,.0f}<br>"
            f"Total: ${scenario_cost:,.2f}<extra></extra>"
        )
    ))

fig.update_layout(
    height=450,
    xaxis_title="Minutes / month",
    yaxis_title="Total cost ($)  (monthly + optional one-time for Neto)",
    legend=dict(font=dict(size=10))
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Scenario comparison (stacked components)
# -----------------------------
st.markdown("### Scenario comparison (stacked components)")

prov_names = list(providers_effective.keys())
fixed_vals = []
var_vals = []
one_time_vals = []
totals = []

for name in prov_names:
    p = providers_effective[name]
    monthly = monthly_cost(M, p)
    total = total_cost_including_one_time(M, name, p)

    if p.get("type") != "livekit":
        fixed = float(p["fixed_monthly"])
        included = float(p.get("included_minutes", 0.0))
        rate = float(p.get("rate_per_min", 0.0))
        over = max(0.0, M - included)
        var = over * rate
    else:
        r = p["rates"]
        fixed = float(p["fixed_monthly"])

        agent_over = max(0.0, M - float(p["included_agent_minutes"]))
        tel_over = max(0.0, M - float(p["included_telephony_minutes"]))

        agent_cost = agent_over * float(r["agent"])
        tel_cost = tel_over * float(r["telephony"])

        inf_rate = livekit_inference_rate(p)
        inf_gross = M * inf_rate
        inf_net = max(0.0, inf_gross - float(p["inference_credits"]))

        var = agent_cost + tel_cost + inf_net

    one_time = NETO_ONBOARDING_ONE_TIME if (include_neto_onboarding and name.startswith("Neto")) else 0.0

    fixed_vals.append(fixed)
    var_vals.append(var)
    one_time_vals.append(one_time)
    totals.append(total)

bar = go.Figure()
bar.add_trace(go.Bar(name="Fixed / Plan (monthly)", x=prov_names, y=fixed_vals, hovertemplate="$%{y:,.2f}<extra></extra>"))
bar.add_trace(go.Bar(name="Usage / Variable (monthly)", x=prov_names, y=var_vals, hovertemplate="$%{y:,.2f}<extra></extra>"))

if include_neto_onboarding:
    bar.add_trace(go.Bar(name="Neto onboarding (one-time)", x=prov_names, y=one_time_vals, hovertemplate="$%{y:,.2f}<extra></extra>"))

bar.update_layout(
    barmode="stack",
    height=420,
    title=f"Stacked cost at {M:,.0f} minutes/month",
    yaxis_title="Cost ($)",
    xaxis_title="Provider",
)
st.plotly_chart(bar, use_container_width=True)

# -----------------------------
# Scenario breakdown (choose a provider) + Waterfall + Audit table
# -----------------------------
st.markdown("## Scenario breakdown (choose a provider)")

provider_names = list(providers_effective.keys())
focus = st.selectbox("Provider", provider_names, index=0)
p_focus = providers_effective[focus]

def breakdown_items(minutes: float, name: str, p: dict):
    if p.get("type") != "livekit":
        fixed = float(p["fixed_monthly"])
        included = float(p.get("included_minutes", 0.0))
        rate = float(p.get("rate_per_min", 0.0))
        over = max(0.0, minutes - included)
        usage_cost = over * rate

        # Optional one-time for Neto (shown as separate waterfall bar if enabled)
        one_time = NETO_ONBOARDING_ONE_TIME if (include_neto_onboarding and name.startswith("Neto")) else 0.0

        items = [("Fixed / Plan fee", fixed), ("Usage (minutes × rate)", usage_cost)]
        if one_time > 0:
            items.append(("One-time onboarding", one_time))

        meta = {
            "minutes": minutes,
            "fixed": fixed,
            "included": included,
            "rate": rate,
            "over": over,
            "usage_cost": usage_cost,
            "one_time": one_time,
            "total": fixed + usage_cost + one_time
        }
        return items, meta

    r = p["rates"]
    fixed = float(p["fixed_monthly"])
    inc_agent = float(p["included_agent_minutes"])
    inc_tel = float(p["included_telephony_minutes"])
    credits = float(p["inference_credits"])

    agent_rate = float(r["agent"])
    tel_rate = float(r["telephony"])
    llm_rate = float(r["llm"])
    stt_rate = float(r["stt"])
    tts_rate = float(r["tts"])
    inf_rate = llm_rate + stt_rate + tts_rate

    agent_over = max(0.0, minutes - inc_agent)
    tel_over = max(0.0, minutes - inc_tel)

    agent_cost = agent_over * agent_rate
    tel_cost = tel_over * tel_rate

    inf_gross = minutes * inf_rate
    credit_used = min(credits, inf_gross)
    inf_net = max(0.0, inf_gross - credits)

    items = [
        ("Plan fee", fixed),
        ("Agent overage", agent_cost),
        ("Telephony overage", tel_cost),
        ("LLM", minutes * llm_rate),
        ("STT", minutes * stt_rate),
        ("TTS", minutes * tts_rate),
        ("Inference credits applied", -credit_used),
    ]
    meta = {
        "minutes": minutes,
        "fixed": fixed,
        "inc_agent": inc_agent,
        "inc_tel": inc_tel,
        "agent_rate": agent_rate,
        "tel_rate": tel_rate,
        "llm_rate": llm_rate,
        "stt_rate": stt_rate,
        "tts_rate": tts_rate,
        "agent_over": agent_over,
        "tel_over": tel_over,
        "agent_cost": agent_cost,
        "tel_cost": tel_cost,
        "inf_gross": inf_gross,
        "credit_used": credit_used,
        "inf_net": inf_net,
        "total": fixed + agent_cost + tel_cost + inf_net
    }
    return items, meta

items, meta = breakdown_items(M, focus, p_focus)

labels = [k for k, _ in items] + ["Total"]
vals = [v for _, v in items]
wf_total = sum(vals)

wf = go.Figure(go.Waterfall(
    orientation="v",
    measure=["relative"] * len(vals) + ["total"],
    x=labels,
    y=vals + [wf_total],
    hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"
))
wf.update_layout(
    height=360,
    title=f"{focus} — breakdown at {M:,.0f} minutes/month",
    yaxis_title="Cost ($)",
)
st.plotly_chart(wf, use_container_width=True)

# Quick summary numbers
c1, c2, c3 = st.columns(3)
c1.metric("Minutes / month", f"{M:,.0f}")
c2.metric("Total (incl optional one-time)", f"${meta['total']:,.2f}")
c3.metric("Effective $/min (model)", f"${(meta['total']/M if M > 0 else 0.0):.4f}")

# Audit table (textual + explicit substitutions)
with st.expander("Show calculation details (audit)"):
    st.markdown("### Inputs used")
    st.write(f"- Minutes/month (M): **{M:,.0f}**")

    if p_focus.get("type") != "livekit":
        fixed = meta["fixed"]
        included = meta["included"]
        rate = meta["rate"]
        over = meta["over"]
        usage_cost = meta["usage_cost"]
        one_time = meta["one_time"]
        total = meta["total"]

        st.markdown("### Parameters")
        st.write(f"- Fixed/plan fee: **${fixed:,.2f}**")
        st.write(f"- Included minutes: **{included:,.0f}**")
        st.write(f"- Rate: **${rate:,.5f}/min**")
        if include_neto_onboarding and focus.startswith("Neto"):
            st.write(f"- One-time onboarding: **${NETO_ONBOARDING_ONE_TIME:,.0f}** (included)")

        calc = pd.DataFrame([
            {
                "Component": "Overage minutes",
                "Formula": "max(0, M − included)",
                "Substitution": f"max(0, {M:,.0f} − {included:,.0f})",
                "Result": f"{over:,.0f} min",
                "Value ($)": f"{usage_cost:,.2f}",
            },
            {
                "Component": "Usage cost",
                "Formula": "overage × rate",
                "Substitution": f"{over:,.0f} × ${rate:,.5f}",
                "Result": f"${usage_cost:,.2f}",
                "Value ($)": f"{usage_cost:,.2f}",
            },
            {
                "Component": "Total (monthly)",
                "Formula": "fixed + usage",
                "Substitution": f"${fixed:,.2f} + ${usage_cost:,.2f}",
                "Result": f"${(fixed + usage_cost):,.2f}",
                "Value ($)": f"{(fixed + usage_cost):,.2f}",
            },
        ])

        if one_time > 0:
            calc = pd.concat([calc, pd.DataFrame([{
                "Component": "One-time onboarding",
                "Formula": "add one-time fee",
                "Substitution": f"+ ${NETO_ONBOARDING_ONE_TIME:,.0f}",
                "Result": f"${one_time:,.2f}",
                "Value ($)": f"{one_time:,.2f}",
            }, {
                "Component": "Total (incl one-time)",
                "Formula": "monthly + one-time",
                "Substitution": f"${(fixed + usage_cost):,.2f} + ${one_time:,.2f}",
                "Result": f"${total:,.2f}",
                "Value ($)": f"{total:,.2f}",
            }])], ignore_index=True)

        st.dataframe(calc, use_container_width=True)

    else:
        # LiveKit audit
        inc_agent = meta["inc_agent"]
        inc_tel = meta["inc_tel"]
        agent_rate = meta["agent_rate"]
        tel_rate = meta["tel_rate"]
        llm_rate = meta["llm_rate"]
        stt_rate = meta["stt_rate"]
        tts_rate = meta["tts_rate"]
        agent_over = meta["agent_over"]
        tel_over = meta["tel_over"]
        agent_cost = meta["agent_cost"]
        tel_cost = meta["tel_cost"]
        inf_gross = meta["inf_gross"]
        credit_used = meta["credit_used"]
        inf_net = meta["inf_net"]
        fixed = meta["fixed"]
        total = meta["total"]

        st.markdown("### Parameters")
        st.write(f"- Plan fee: **${fixed:,.2f}**")
        st.write(f"- Included agent minutes: **{inc_agent:,.0f}** at **${agent_rate:.4f}/min**")
        st.write(f"- Included telephony minutes: **{inc_tel:,.0f}** at **${tel_rate:.4f}/min**")
        st.write(f"- LLM: **${llm_rate:.4f}/min**, STT: **${stt_rate:.4f}/min**, TTS: **${tts_rate:.4f}/min**")
        st.write(f"- Inference credits: **${p_focus['inference_credits']:,.2f}**")

        calc = pd.DataFrame([
            {
                "Component": "Agent overage minutes",
                "Formula": "max(0, M − included_agent)",
                "Substitution": f"max(0, {M:,.0f} − {inc_agent:,.0f})",
                "Result": f"{agent_over:,.0f} min",
                "Value ($)": f"{agent_cost:,.2f}",
            },
            {
                "Component": "Telephony overage minutes",
                "Formula": "max(0, M − included_tel)",
                "Substitution": f"max(0, {M:,.0f} − {inc_tel:,.0f})",
                "Result": f"{tel_over:,.0f} min",
                "Value ($)": f"{tel_cost:,.2f}",
            },
            {
                "Component": "LLM cost",
                "Formula": "M × llm_rate",
                "Substitution": f"{M:,.0f} × ${llm_rate:.4f}",
                "Result": f"${(M*llm_rate):,.2f}",
                "Value ($)": f"{(M*llm_rate):,.2f}",
            },
            {
                "Component": "STT cost",
                "Formula": "M × stt_rate",
                "Substitution": f"{M:,.0f} × ${stt_rate:.4f}",
                "Result": f"${(M*stt_rate):,.2f}",
                "Value ($)": f"{(M*stt_rate):,.2f}",
            },
            {
                "Component": "TTS cost",
                "Formula": "M × tts_rate",
                "Substitution": f"{M:,.0f} × ${tts_rate:.4f}",
                "Result": f"${(M*tts_rate):,.2f}",
                "Value ($)": f"{(M*tts_rate):,.2f}",
            },
            {
                "Component": "Inference gross",
                "Formula": "LLM + STT + TTS",
                "Substitution": f"${(M*llm_rate):,.2f} + ${(M*stt_rate):,.2f} + ${(M*tts_rate):,.2f}",
                "Result": f"${inf_gross:,.2f}",
                "Value ($)": f"{inf_gross:,.2f}",
            },
            {
                "Component": "Inference credits applied",
                "Formula": "min(credits, inference_gross)",
                "Substitution": f"min(${p_focus['inference_credits']:,.2f}, ${inf_gross:,.2f})",
                "Result": f"-${credit_used:,.2f}",
                "Value ($)": f"-{credit_used:,.2f}",
            },
            {
                "Component": "Inference net",
                "Formula": "max(0, inference_gross − credits)",
                "Substitution": f"max(0, ${inf_gross:,.2f} − ${p_focus['inference_credits']:,.2f})",
                "Result": f"${inf_net:,.2f}",
                "Value ($)": f"{inf_net:,.2f}",
            },
            {
                "Component": "Total (monthly)",
                "Formula": "plan + agent + tel + inference_net",
                "Substitution": f"${fixed:,.2f} + ${agent_cost:,.2f} + ${tel_cost:,.2f} + ${inf_net:,.2f}",
                "Result": f"${total:,.2f}",
                "Value ($)": f"{total:,.2f}",
            },
        ])
        st.dataframe(calc, use_container_width=True)

