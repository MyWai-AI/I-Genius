"""MyWai authentication module."""

import logging
from typing import Optional

import requests
import streamlit as st
from mywai_python_integration_kit.apis import initialize_apis
from mywai_python_integration_kit.apis.services.users_api import login, get_users
from mywai_python_integration_kit.data.mywai_schemas.base_apis import UserLoginCredentials
from mywai_python_integration_kit.utils.system import check_str_docker_local_execution
from streamlit_post_message.streamlit_post_message import streamlit_post_message

from src.streamlit_template.core.Common.config import Config

logger = logging.getLogger(__name__)


def debug_login_form() -> Optional[dict]:
    """
    Shows a login form when DEBUG_MODE is True.
    Returns the MyWai payload if login is successful, None otherwise.
    """
    st.title("MYWAI Login")


    with st.form("debug_login_form"):
        email = st.text_input("Email", value="your Email", placeholder="example@myw.ai")
        password = st.text_input("Password", type="password", value="your Password", placeholder="Enter your password")
        endpoint = st.text_input(
            "MYWAI Endpoint",
            #value=Config.MYWAI_ENDPOINT or "https://localhost:44330",
            value=Config.MYWAI_ENDPOINT or "https://igenius.platform.myw.ai/api",
            #placeholder="https://localhost:44330",
            placeholder="https://igenius.platform.myw.ai/api",
            help="MyWai platform endpoint",
        )

        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email or not password or not endpoint:
                st.error("Please fill in all fields")
                return None

            # Normalize endpoint (remove trailing slash)
            endpoint = endpoint.rstrip("/")

            try:
                user_credential = UserLoginCredentials(email=email, password=password)
                auth_token, _ = login(user_credential)
                
                if auth_token is not None:
                    # Create a mock payload similar to streamlit_post_message
                    mywai_payload = {
                        "user": {
                            "id": None,
                            "name": None,
                            "surname": None,
                            "email": email,
                            "auth-token": auth_token,
                        },
                        "config": {
                            "api-endpoint": endpoint,
                        },
                    }

                    # Save credentials in session state for reuse
                    st.session_state["debug_auth"] = {
                        "email": email,
                        "endpoint": endpoint,
                        "mywai_payload": mywai_payload,
                    }

                    logger.info("Debug login successful for user %s", email)
                    st.success("Login successful!")
                    return mywai_payload
                else:
                    st.error("Login error: AuthToken is None")
                    return None

            except requests.exceptions.RequestException as e:
                logger.exception("Connection error during debug login")
                st.error(f"Connection error: {str(e)}")
                return None
            except Exception as e:
                logger.exception("Unexpected error during debug login")
                st.error(f"Unexpected error: {str(e)}")
                return None

    return None


def get_mywai_payload() -> Optional[dict]:
    """
    Get MyWai payload based on current mode (debug or normal).
    Returns the payload if authentication is successful, None otherwise.
    """
    if Config.DEBUG_MODE:
        # Check if we already logged in
        if "debug_auth" not in st.session_state:
            mywai_payload = debug_login_form()
            if mywai_payload is None:
                return None  # Show only the login form
            st.rerun()
        else:
            mywai_payload = st.session_state["debug_auth"]["mywai_payload"]
    else:
        # Normal mode: use streamlit_post_message
        mywai_payload = streamlit_post_message(
            sleep_time=1.0, message_key="mywai-tool-init"
        )

    return mywai_payload


def initialize_mywai_apis(mywai_payload: dict) -> str:
    """
    Initialize MyWai APIs with the provided payload.
    Returns the MyWai endpoint used for initialization.
    """
    if Config.MYWAI_ENDPOINT is None:
        mywai_endpoint = check_str_docker_local_execution(mywai_payload["config"]["api-endpoint"])
    else:
        mywai_endpoint = check_str_docker_local_execution(Config.MYWAI_ENDPOINT)
    
    auth_token = mywai_payload["user"]["auth-token"]
    initialize_apis(endpoint=mywai_endpoint, auth_token=auth_token)
    
    return mywai_endpoint


def enrich_user_data(mywai_payload: dict) -> None:
    """
    Enrich user data in debug mode by fetching additional user information.
    """
    if Config.DEBUG_MODE:
        try:
            users = get_users()
            for u in users:
                if u["email"] == mywai_payload["user"]["email"]:
                    mywai_payload["user"]["id"] = u["userId"]
                    mywai_payload["user"]["surname"] = u["surname"]
                    mywai_payload["user"]["name"] = u["name"]
                    break
        except Exception as e:
            logger.warning("Failed to enrich user data: %s", str(e))


def logout():
    """Logout user and clear session state."""
    if "debug_auth" in st.session_state:
        del st.session_state["debug_auth"]
    st.rerun()
