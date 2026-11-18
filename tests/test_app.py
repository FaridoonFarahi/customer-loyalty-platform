"""Every page of the Streamlit app must render without an error."""
import pytest

from src import config

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

PAGES = [
    "Overview", "Who to target", "Score a household",
    "Plan an experiment", "Campaign effect and uplift", "Model and pipeline",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    app = streamlit_testing.AppTest.from_file(str(config.PROJECT_ROOT / "streamlit_app.py"), default_timeout=60)
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert app.title[0].value
