"""Minimal Streamlit user interface for survey-item generation."""

from __future__ import annotations


def main() -> None:
    """Collect generation settings and present the generated survey items."""
    import streamlit as st

    st.set_page_config(page_title="Survey Item Generator", page_icon="📝")
    st.title("Survey Item Generator")
    st.caption("Generate survey items from a research question and target group.")

    with st.sidebar:
        research_question = st.text_area("Research question")
        target_group = st.text_input("Target group")
        requirements = st.text_area("Additional requirements")
        n_items = st.number_input(
            "Number of items",
            min_value=1,
            max_value=50,
            value=10,
        )
        generate = st.button("Generate items", type="primary")

    if generate:
        if not research_question or not target_group:
            st.warning("Please enter a research question and target group.")
            st.stop()

        # These values are passed to the retrieval and generation functions.
        generation_request = {
            "research_question": research_question,
            "target_group": target_group,
            "requirements": requirements,
            "n_items": int(n_items),
        }
        st.success("Generation request submitted.")
        st.json(generation_request)

        # In the complete application, the generated DataFrame is displayed as:
        # st.dataframe(generated_items, use_container_width=True)
        # st.download_button(
        #     "Download CSV",
        #     generated_items.to_csv(index=False),
        #     "generated_items.csv",
        # )


if __name__ == "__main__":
    main()
