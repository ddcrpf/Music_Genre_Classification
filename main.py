#app.py
import app1
import app2
import streamlit as st

st.set_page_config(page_title="My Streamlit App", page_icon="🎤", layout="wide")

PAGES = {
    "🧐 Predict": app1,
    "👋 About": app2
}

title1 = """
<p style = "
font-size: 40px;">
🧭 Navigation</p>
"""

st.sidebar.markdown(title1,unsafe_allow_html=True)
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.main()