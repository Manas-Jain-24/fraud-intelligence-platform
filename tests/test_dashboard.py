"""Exercise the real Streamlit app with an isolated persistent investigation store."""
from streamlit.testing.v1 import AppTest
from pathlib import Path
from fraud_ops.store import InvestigationStore


def test_dashboard_demo_assignment_resolution_and_rail_switch():
    at = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not at.exception
    if (Path(__file__).resolve().parents[1] / "data/artifacts/workflow/report.json").exists():
        assert any("Whole-workflow evaluation" in expander.label for expander in at.expander)
        assert any("Staffing sensitivity" in item.value for item in at.markdown)
    at.sidebar.toggle[0].set_value(False)
    at.button(key="ops_seed_demo").click().run()
    assert not at.exception
    assert at.selectbox(key="ops_selected_case").options
    case_id = at.selectbox(key="ops_selected_case").value
    at.text_input(key="ops_actor").set_value("Asha")
    assignee = next(widget for widget in at.text_input if widget.label == "Assignee")
    assignee.set_value("Asha")
    save = next(widget for widget in at.button if widget.label == "Save case action")
    save.click().run()
    assert not at.exception
    assert InvestigationStore().get_case(case_id)["status"] == "IN_REVIEW"
    action = next(widget for widget in at.selectbox if widget.label == "Case action")
    action.set_value("RESOLVE")
    reason = next(widget for widget in at.text_area if widget.label == "Reason / investigation note")
    reason.set_value("Verified synthetic investigation workflow")
    next(widget for widget in at.button if widget.label == "Save case action").click().run()
    assert not at.exception
    assert InvestigationStore().get_case(case_id)["decision"] == "CONFIRMED_FRAUD"
    assert InvestigationStore().feedback_rows() == []
    assert len(InvestigationStore().feedback_rows(include_synthetic=True)) == 1
    at.sidebar.radio[0].set_value("📱 Mobile Money Rail (PaySim - Ledger Drain)").run()
    assert not at.exception
    assert any("post-transaction balances" in info.value for info in at.info)
    # A research-model selection must change the actual sandbox scorer/explanation.
    next(widget for widget in at.selectbox if widget.label == "Sandbox scoring model:").set_value("Cost-Sensitive LogReg (NumPy)").run()
    assert not at.exception
    assert any("Cost-Sensitive LogReg (NumPy)" in info.value for info in at.info)
