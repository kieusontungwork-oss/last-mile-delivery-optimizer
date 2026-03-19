"""Streamlit entry point for the Last-Mile Delivery Optimizer."""

import streamlit as st

st.set_page_config(
    page_title="Last-Mile Delivery Optimizer",
    layout="wide",
)

st.title("Last-Mile Delivery Optimizer")
st.caption("ML-driven dynamic routing for last-mile delivery optimization")

page = st.sidebar.selectbox("Navigate", ["Optimize", "Compare", "Dashboard"])

if page == "Optimize":
    from frontend.pages.optimize import render
    render()
elif page == "Compare":
    from frontend.pages.compare import render
    render()
elif page == "Dashboard":
    from frontend.pages.dashboard import render
    render()
