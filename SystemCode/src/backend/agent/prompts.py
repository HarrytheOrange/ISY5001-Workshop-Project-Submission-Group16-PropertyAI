# prompt.py

def get_user_need_prompt(user_needs: dict) -> str:
    """
    Returns the system prompt for the chatbot, given the current user_needs.
    """
    return f"""
            You are a helpful assistant collecting user housing needs.
            Current known user_needs:
            {user_needs}
            Ask for missing ones. Each time, ask for only one missing field.
            If you learn a new value, call the update_user_need tool.
            """
