import streamlit as st

def render_agent():
    st.markdown("---")

    # greeting
    st.markdown(
        '<div class="agent-greeting">'
        '<h3>Rini — AI Business Consultant</h3>'
        '<p>Halo! Saya <b>Rini</b>, AI Business Consultant Nusantara Connect. '
        'Saya bisa membantu Anda:</p>'
        '<p>'
        '<b>Menjawab pertanyaan</b> tentang perusahaan &amp; layanan<br>'
        '<b>Memprediksi churn</b> pelanggan secara langsung<br>'
        '<b>Menganalisis data</b> pelanggan (594K records)<br>'
        '<b>Membuat visualisasi</b> chart interaktif'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # session state init
    if "agent_history"not in st.session_state:
        st.session_state["agent_history"] = []  # [{role, content, charts?}]
    if "agent_llm_history"not in st.session_state:
        st.session_state["agent_llm_history"] = []  # clean role/content for API

    # quick prompts
    st.markdown('<p style="color:#64748b; font-size:13px; margin-bottom:4px;">Contoh pertanyaan:</p>', unsafe_allow_html=True)
    qp_cols = st.columns(4)
    quick_prompts = [
        "Berapa churn rate pelanggan Fiber optic?",
        "Prediksi churn: Female, tenure 3, Fiber optic, Month-to-month",
        "Buatkan grafik churn rate by contract type",
        "Apa layanan utama Nusantara Connect?",
    ]
    selected_prompt = None
    for i, qp in enumerate(quick_prompts):
        with qp_cols[i]:
            st.markdown('<div class="quick-prompt-btn">', unsafe_allow_html=True)
            if st.button(qp, key=f"qp_{i}", width='stretch'):
                selected_prompt = qp
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # render chat history
    for msg in st.session_state["agent_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # render charts if present
            if msg.get("charts"):
                for fig in msg["charts"]:
                    st.plotly_chart(fig, width='stretch')

    # chat input
    user_input = st.chat_input("Tanya Rini tentang data, prediksi, atau analisis...", key="agent_chat_input")

    # use quick prompt if selected
    if selected_prompt:
        user_input = selected_prompt

    if user_input:
        # display user message
        st.session_state["agent_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # call agent
        with st.chat_message("assistant"):
            with st.spinner("Rini sedang berpikir..."):
                try:
                    from llm.agent import run_agent
                    response_text, charts = run_agent(
                        user_input,
                        st.session_state["agent_llm_history"],
                    )
                except Exception as e:
                    response_text = f"Terjadi kesalahan: {e}"
                    charts = []

            st.markdown(response_text)
            for fig in charts:
                st.plotly_chart(fig, width='stretch')

        # save to history
        st.session_state["agent_history"].append({
            "role": "assistant",
            "content": response_text,
            "charts": charts,
        })
        # LLM history (no charts, just text)
        st.session_state["agent_llm_history"].append({"role": "user", "content": user_input})
        st.session_state["agent_llm_history"].append({"role": "assistant", "content": response_text})

        # keep history manageable (last 20 turns)
        if len(st.session_state["agent_llm_history"]) > 40:
            st.session_state["agent_llm_history"] = st.session_state["agent_llm_history"][-40:]

        st.rerun()

    # clear chat button
    if st.session_state["agent_history"]:
        st.markdown("---")
        if st.button("Clear Chat", key="clear_agent_chat", width='stretch'):
            st.session_state["agent_history"] = []
            st.session_state["agent_llm_history"] = []
            st.rerun()
