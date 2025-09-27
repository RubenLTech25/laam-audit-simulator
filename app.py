import streamlit as st
import math

# --- LAAM Core Functions ---

def calculate_anchor_decay(initial_weight: float, decay_rate: float, time_elapsed: int) -> float:
    return initial_weight * math.exp(-decay_rate * time_elapsed)

def adjust_anchor_weight(base_weight: float, recurrence_count: int) -> float:
    recurrence_penalty = [1.0, 1.5, 2.0]
    penalty_index = min(recurrence_count - 1, len(recurrence_penalty) - 1)
    return base_weight * recurrence_penalty[penalty_index]

def apply_forgiveness(anchors: list, clean_cycles: int, threshold: int = 2) -> list:
    if clean_cycles >= threshold:
        return [a * 0.5 for a in anchors]
    else:
        return anchors

def has_compliance_upgrades(entity: dict) -> bool:
    return entity.get("compliance_upgrades", False)

def enforce_fairness(audit_scope: str, anchors: list, entity: dict) -> str:
    GRIM_THRESHOLD = 2.5
    if sum(anchors) > GRIM_THRESHOLD:
        return "Full audit + penalty review"
    elif has_compliance_upgrades(entity):
        return "Limited review"
    else:
        return audit_scope

# --- Bayesian Auditor Class ---

def normalize(prob_dict: dict) -> dict:
    total = sum(prob_dict.values())
    return {k: v / total for k, v in prob_dict.items()}

class BayesianAuditor:
    def __init__(self, sector: str):
        self.sector_priors = {
            'tax': {'C': 0.6, 'R': 0.3, 'E': 0.1}
        }
        self.beliefs = self.sector_priors.get(sector, {'C': 0.6, 'R': 0.3, 'E': 0.1}).copy()
        self.signal_likelihood = {
            'clean': {'C': 0.9, 'R': 0.4, 'E': 0.05},
            'vague': {'C': 0.05, 'R': 0.4, 'E': 0.3},
            'missing_doc': {'C': 0.02, 'R': 0.3, 'E': 0.3},  # Adjusted per LAAM
            'unsafe_condition': {'C': 0.0, 'R': 0.5, 'E': 0.2}
        }

    def update_beliefs(self, signal: str):
        posterior_numerator = {}
        for entity_type in self.beliefs:
            posterior_numerator[entity_type] = self.signal_likelihood[signal][entity_type] * self.beliefs[entity_type]
        self.beliefs = normalize(posterior_numerator)

# --- Streamlit App ---

st.title("LAAM Audit Simulator")

st.subheader("⚖️ Jurisdiction Settings")
jurisdiction = st.selectbox("Choose jurisdiction:", ["IRS (default)", "Florida", "Chicago", "Nevada"])

# Include your reset logic here
if "last_jurisdiction" not in st.session_state:
    st.session_state.last_jurisdiction = jurisdiction

if jurisdiction != st.session_state.last_jurisdiction:
    st.session_state.anchors = []
    st.session_state.anchor_history = []
    st.session_state.beliefs = [0.33, 0.33, 0.34]
    st.session_state.belief_history = []
    st.session_state.audit_log = []
    st.session_state.anchor_weights = []
    st.session_state.signal_history = []
    st.session_state.cycles = 0
    st.session_state.clean_cycles = 0
    st.session_state.final_scope = "Not yet evaluated"
    st.session_state.override = False
    st.session_state.decayed_anchors = []
    st.session_state.forgiven_anchors = []
    st.session_state.legacy_score = 0
    st.session_state.last_jurisdiction = jurisdiction

st.markdown("""
### 🚀 Start Here: Quick Walkthrough

Welcome to the LAAM Audit Simulator. Here's how to begin:

1. **Choose a jurisdiction** using the dropdown above.
2. **Start a new audit cycle** by clicking the button.
3. **Select a signal** that reflects the taxpayer’s behavior.
4. **Assign anchors** if the signal is risky.
5. **Run the LAAM Protocol** to update beliefs and audit scope.
6. Watch how **belief drift and anchor burden** evolve over time.

After 3 cycles, you'll receive a narrative summary of the audit journey so far.
""")

with st.expander("📘 How This App Works"):
    st.markdown("""
    **Welcome to the LAAM Audit Simulator.**  
    This app lets you explore how audit decisions evolve over time using fairness-aware logic.

    ### 🌀 What You’re Simulating
    - You play as an auditor reviewing a taxpayer over multiple cycles.
    - Each cycle, you receive a signal (e.g., clean, vague, missing documentation).
    - Based on the signal, the system updates its beliefs about the taxpayer’s behavior.

    ### 🧠 What LAAM Does
    - Assigns anchors (weights) to risky signals.
    - Applies decay and forgiveness over time.
    - Adjusts audit scope based on total burden and compliance upgrades.

    ### 🔄 How to Play
    1. **Start a new audit cycle**
    2. **Choose a signal** that reflects the taxpayer’s behavior
    3. **Update beliefs** to see how the system classifies them
    4. **Assign anchors** if the signal is risky
    5. **Run LAAM Protocol** to apply fairness logic
    6. After 3 clean cycles, audits may be skipped automatically

    ### 🧑‍💼 Manager Override
    You can manually skip an audit to simulate human discretion.

    ---
    This simulation is based on real-world audit logic and Ruben Lopez’s LAAM framework. It’s designed to explore how fairness, memory, and redemption can reshape enforcement systems.
    """)

    st.sidebar.title("🧭 LAAM Reference")

    with st.sidebar.expander("📚 Glossary & Game Terms", expanded=True):
        st.markdown("""
        ### Glossary & Game Terms

        ### 🧠 Core Concepts

        - **Anchor**: A weight assigned to risky behavior. The more anchors an entity accumulates, the more likely they are to face a full audit.
        - **Decay**: Anchors fade over time using exponential decay. Faster decay means quicker forgiveness.
        - **Forgiveness**: If an entity has enough clean cycles, their anchor burden is reduced further.
        - **Clean Cycle**: An audit cycle with no violations. Clean cycles build trust and can trigger audit skips.
        - **Legacy Score**: A cumulative risk metric used in Nevada. Even low anchor burden can trigger audits if legacy score is high.
        - **Override**: Manual audit skip to simulate human discretion—especially useful for high-visibility entities.

        ### 🌀 Audit Signals

        - **`clean`**: No issues found. Builds trust and contributes to forgiveness.
        - **`vague`**: Documentation is unclear. May trigger moderate anchor weight.
        - **`missing_doc`**: Required documentation is absent. Often triggers anchor assignment.
        - **`unsafe_condition`**: Environmental or procedural risk. High likelihood of escalation.
        - **`resolved_error`** *(Florida only)*: A past violation that was corrected. Starts with high anchor but decays quickly.
        - **`legacy_flag`** *(Chicago only)*: A lingering penalty from past infractions. Slow to decay, hard to forgive.
        - **`environmental_trigger`** *(Nevada only)*: Systemic risk signal. Moderate anchor, but contributes heavily to legacy score.

        ---
        These terms reflect real-world audit logic and Ruben Lopez’s LAAM framework. The simulation is designed to explore how fairness, memory, and discretion shape enforcement systems.
        """)

        with st.sidebar.expander("🧭 Scenario Guide", expanded=False):
            st.markdown("""
            **What is a Scenario?**  
            A scenario defines the jurisdiction, signal logic, and audit behavior. Each region interprets signals differently.

                   ### IRS (Default)
                   - **Audit Type**: R&D Credit Disputes  
                   - **Tone**: Technical, high-stakes  
                   - **Behavior**: Balanced enforcement using Bayesian classification and anchor logic. Signals like `missing_doc` and `vague` shape belief updates. Forgiveness after 2 clean cycles.

                   ### Florida
                   - **Audit Type**: Sales Tax  
                   - **Tone**: Fast-paced, corrective  
                   - **Behavior**: Resolved errors start with high anchor (5.0) but decay rapidly. Forgiveness is generous. Anchors fade quickly after clean cycles.

                   ### Chicago
                   - **Audit Type**: Parking Tax  
                   - **Tone**: Bureaucratic, legacy-heavy  
                   - **Behavior**: Slow forgiveness (requires 4 clean cycles). Legacy flags linger. Anchors decay slowly. Audit scope rarely downgrades without upgrades.

                   ### Nevada
                   - **Audit Type**: No Income Tax (False Positives)  
                   - **Tone**: Risk-reactive, systemic  
                   - **Behavior**: Environmental triggers dominate. Legacy score can override forgiveness. Even low anchor burden may trigger full audits if systemic risk is high.

                   ---
                   Each jurisdiction reflects a different philosophy of enforcement. Use this guide to explore how fairness, memory, and discretion shape audit outcomes.
                   """)


    # Initialize tracker if missing
    if "last_jurisdiction" not in st.session_state:
        st.session_state.last_jurisdiction = jurisdiction

    # Reset if jurisdiction changes
    if jurisdiction != st.session_state.last_jurisdiction:
        st.session_state.anchors = []
        st.session_state.anchor_history = []
        st.session_state.beliefs = [0.33, 0.33, 0.34]
        st.session_state.belief_history = []
        st.session_state.audit_log = []
        st.session_state.anchor_weights = []
        st.session_state.signal_history = []
        st.session_state.cycles = 0
        st.session_state.clean_cycles = 0
        st.session_state.final_scope = "Not yet evaluated"
        st.session_state.override = False
        st.session_state.decayed_anchors = []
        st.session_state.forgiven_anchors = []
        st.session_state.legacy_score = 0

        # Update tracker
        st.session_state.last_jurisdiction = jurisdiction

st.write("Walk through the audit cycle, experience asymmetric enforcement, and apply the Lopez Model.")

# Initialize session state
if "auditor" not in st.session_state:
    st.session_state.auditor = BayesianAuditor(sector="tax")
    st.session_state.anchors = []
    st.session_state.entity = {"compliance_upgrades": False}
    st.session_state.clean_cycles = 2

if "beliefs" not in st.session_state:
    st.session_state.beliefs = [0.33, 0.33, 0.34]  # Default neutral belief scores for C, R, E

def update_beliefs(signal, current_beliefs, jurisdiction):
    # Base drift weights per signal
    drift_map = {
        "clean": [-0.05, -0.05, -0.05],
        "vague": [0.0, 0.05, 0.1],
        "missing_doc": [0.0, 0.1, 0.2],
        "resolved_error": [-0.1, -0.05, -0.05],
        "legacy_flag": [0.0, 0.1, 0.15],
        "environmental_trigger": [0.0, 0.05, 0.2]
    }

    drift = drift_map.get(signal, [0.0, 0.0, 0.0])

    # Jurisdictional tuning (optional)
    if jurisdiction == "Chicago" and signal == "legacy_flag":
        drift = [0.0, 0.15, 0.25]
    elif jurisdiction == "Florida" and signal == "resolved_error":
        drift = [-0.2, -0.1, -0.1]

    if jurisdiction == "IRS (default)":
        if signal == "missing_doc":
            drift = [0.0, 0.15, 0.1]
        elif signal == "unsafe_condition":
            drift = [0.0, 0.2, 0.15]
        elif signal == "clean":
            drift = [-0.1, -0.05, -0.05]

    if jurisdiction == "IRS (default)" and sum(st.session_state.anchors) > 4 and st.session_state.beliefs[0] < 0.15:
        st.session_state.override = True
        st.session_state.final_scope = "Override triggered: systemic risk"

    if st.session_state.cycles > len(st.session_state.anchor_history):
        burden = round(sum(st.session_state.forgiven_anchors), 2)
        st.session_state.anchor_history.append(burden)

    # Apply drift
    updated = [max(0.0, min(1.0, b + d)) for b, d in zip(current_beliefs, drift)]
    total = sum(updated)
    return [b / total for b in updated] if total > 0 else [0.33, 0.33, 0.34]

if "cycles" not in st.session_state:
    st.session_state.cycles = 0

if "final_scope" not in st.session_state:
    st.session_state.final_scope = "Not yet evaluated"

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "anchor_history" not in st.session_state:
    st.session_state.anchor_history = []

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

if "anchor_weights" not in st.session_state:
    st.session_state.anchor_weights = []

if "belief_history" not in st.session_state:
    st.session_state.belief_history = []

if "last_jurisdiction" not in st.session_state:
    st.session_state.last_jurisdiction = "IRS (default)"

# Step 1: Signal Input
signal_descriptions = {
    "clean": "No issues found",
    "vague": "Unclear documentation",
    "missing_doc": "Required documentation missing",
    "unsafe_condition": "Environmental or procedural risk",
    "resolved_error": "Past violation corrected (Florida)",
    "legacy_flag": "Lingering penalty from past infractions (Chicago)",
    "environmental_trigger": "Systemic risk signal (Nevada)"
}

# Filter available signals based on jurisdiction
available_signals = ["clean", "vague", "missing_doc", "unsafe_condition"]
if jurisdiction == "Florida":
    available_signals.append("resolved_error")
elif jurisdiction == "Chicago":
    available_signals.append("legacy_flag")
elif jurisdiction == "Nevada":
    available_signals.append("environmental_trigger")

# Display signal with description
signal = st.selectbox(
    "Choose the audit signal received:",
    available_signals,
    format_func=lambda s: f"{s} – {signal_descriptions[s]}"
)

# Step 2: Assign Anchor
if st.button("Assign Anchor"):
    if signal == "resolved_error":
        anchor_weight = 5.0
    elif signal == "legacy_flag" and jurisdiction == "Chicago":
        anchor_weight = 2.5
    elif signal == "environmental_trigger" and jurisdiction == "Nevada":
        anchor_weight = 1.5
    else:
        recurrence_count = st.session_state.anchors.count(signal) + 1
        anchor_weight = adjust_anchor_weight(base_weight=1.0, recurrence_count=recurrence_count)

    st.session_state.anchors.append(anchor_weight)
    st.session_state.signal_history.append(signal)
    st.session_state.anchor_weights.append((signal, anchor_weight))

    legacy_score = sum(st.session_state.anchors) * 0.3 if jurisdiction == "Nevada" else 0
    st.session_state.legacy_score = legacy_score

    st.write(f"Anchor Assigned: {anchor_weight:.2f}")
    st.write(f"Total Anchor Burden: {sum(st.session_state.anchors):.2f}")

# Step 3: Apply LAAM Protocol
if st.button("Run LAAM Protocol"):
    decay_rate = 0.5 if jurisdiction == "Florida" else 0.1 if jurisdiction == "Chicago" else 0.2 if jurisdiction == "Nevada" else 0.3
    forgiveness_threshold = 4 if jurisdiction == "Chicago" else 2

    decayed = [calculate_anchor_decay(a, decay_rate=decay_rate, time_elapsed=2) for a in st.session_state.anchors]
    forgiven = apply_forgiveness(decayed, st.session_state.clean_cycles, threshold=forgiveness_threshold)

    legacy_score = sum(st.session_state.anchors) * 0.3 if jurisdiction == "Nevada" else 0
    final_scope = enforce_fairness("full", forgiven, st.session_state.entity)

    # Escalation override for Nevada
    if jurisdiction == "Nevada" and legacy_score > 1.5:
        final_scope = "Full audit triggered by legacy score"

    st.session_state.final_scope = final_scope
    st.session_state.decayed_anchors = decayed
    st.session_state.forgiven_anchors = forgiven
    st.session_state.legacy_score = legacy_score

    st.session_state.beliefs = update_beliefs(signal, st.session_state.beliefs, jurisdiction)
    st.session_state.belief_history.append(st.session_state.beliefs.copy())

    # --- Narrative Summary After 3 Cycles ---
    if st.session_state.cycles >= 3 and "narrative_shown" not in st.session_state:
        recent_signals = st.session_state.signal_history[-3:]
        compliant_score = st.session_state.beliefs[0]
        scope = st.session_state.final_scope
        jurisdiction = st.session_state.last_jurisdiction

        summary = f"""
        Over the last 3 cycles in **{jurisdiction}**, the system observed signals like {', '.join(recent_signals)}.
        The current belief in compliance is **{round(compliant_score, 2)}**.
        Audit scope has escalated to: **{scope}**.
        """

        st.markdown("### 📖 Audit Narrative Summary")
        st.markdown(summary)

        st.session_state.narrative_shown = True

    # --- Persistent Audit Narrative ---
    if st.session_state.cycles >= 1:
        recent_signals = st.session_state.signal_history[-3:]
        compliant_score = st.session_state.beliefs[0]
        scope = st.session_state.final_scope
        jurisdiction = st.session_state.last_jurisdiction

        st.markdown("### 📖 Audit Narrative Summary")
        st.markdown(f"""
        Over the last {min(3, st.session_state.cycles)} cycles in **{jurisdiction}**,  
        the system observed signals like: `{', '.join(recent_signals)}`.  
        Current belief in compliance: **{round(compliant_score, 2)}**  
        Audit scope: **{scope}**
        """)

    # Visual feedback
    st.write("Anchors After Decay:", [round(a, 2) for a in decayed])
    st.write("Anchors After Forgiveness:", [round(a, 2) for a in forgiven])
    st.write("Legacy Score:", round(legacy_score, 2))
    st.write("Final Audit Scope:", final_scope)

    if "downgraded" in final_scope:
        st.success("Fairness Restored. Audit scope downgraded.")
    elif "triggered" in final_scope:
        st.warning("Legacy risk triggered full audit.")
    else:
        st.error("Systemic Burden. Full audit remains.")


# --- Multi-Cycle Simulation Additions ---

# Initialize cycle state
if "cycles" not in st.session_state:
    st.session_state.cycles = 0

# Start new cycle
if st.button("Start New Audit Cycle"):
    st.session_state.cycles += 1
    st.write(f"🌀 Cycle {st.session_state.cycles} started.")

    # Track clean cycles
    if signal == "clean":
        st.session_state.clean_cycles += 1
        st.write(f"✅ Clean cycle recorded. Total clean cycles: {st.session_state.clean_cycles}")
    else:
        st.session_state.clean_cycles = 0
        st.write("⚠️ Flagged cycle. Clean cycle count reset.")

    # Audit skip logic
    if st.session_state.clean_cycles >= 3:
        st.write("🛑 Audit skipped this cycle. Trust threshold met.")
    else:
        st.write("🔍 Audit required this cycle.")

    # Manager override option
    override = st.checkbox("Override audit decision?")
    st.session_state.override = override  # Track override for logs

    if override:
        st.write("🧑‍💼 Audit manually skipped. Manager discretion applied.")

    st.subheader("📊 Visual Feedback")

    # --- Anchor Burden Chart ---
    if "anchor_history" not in st.session_state:
        st.session_state.anchor_history = []

    # Append only once per cycle
    if st.session_state.cycles > len(st.session_state.anchor_history):
        burden = round(sum(st.session_state.forgiven_anchors), 2)
        st.session_state.anchor_history.append(burden)

    st.subheader("📈 Anchor Burden Over Time")
    st.line_chart(st.session_state.anchor_history, use_container_width=True)

    # --- Belief Drift Over Time ---
    if st.session_state.belief_history:
        belief_labels = ["Compliant", "Risky", "Evasive"]
        belief_data = {label: [] for label in belief_labels}
        annotations = []
        belief_sets = st.session_state.belief_history

        for i in range(1, len(belief_sets)):
            prev = belief_sets[i - 1]
            curr = belief_sets[i]
            delta = [round(c - p, 2) for c, p in zip(curr, prev)]
            if delta[2] > 0.1:
                annotations.append(f"Cycle {i + 1}: Evasive spike (+{delta[2]})")
            elif delta[0] > 0.1:
                annotations.append(f"Cycle {i + 1}: Compliance restored (+{delta[0]})")

        for belief_set in belief_sets:
            for i, label in enumerate(belief_labels):
                belief_data[label].append(round(belief_set[i], 2))

        st.subheader("📉 Belief Drift Over Time")
        st.line_chart(belief_data, use_container_width=True)

        if annotations:
            st.markdown("**🧠 Belief Drift Highlights:**")
            for note in annotations:
                st.write(f"• {note}")
    else:
        st.info("Belief drift will appear here once cycles begin.")

    # --- Audit Log Table ---
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []

    audit_entry = {
        "Cycle": st.session_state.cycles,
        "Jurisdiction": jurisdiction,
        "Signal": signal,
        "Anchor Burden": round(sum(st.session_state.anchors), 2),
        "Audit Scope": st.session_state.final_scope,
        "Clean Cycles": st.session_state.clean_cycles,
        "Override": st.session_state.override,
        "Beliefs": st.session_state.beliefs.copy()
    }

    if len(st.session_state.audit_log) < st.session_state.cycles:
        st.session_state.audit_log.append(audit_entry)

    st.dataframe(st.session_state.audit_log, use_container_width=True)

CMD ["streamlit", "run", "laam_simulator.py", "--server.port", "7860", "--server.address", "0.0.0.0"]

    
