"""Helper for issuing the hidden-participant token the saa hosted
agent uses to join a LiveKit room.

The customer's backend calls `attention_agent_token(...)` with THEIR LiveKit
API key+secret and hands the resulting JWT to `start_attention_session(...)`.
We never see the customer's LK API key — only the minimum-grant token they
issued specifically for our agent.
"""
from __future__ import annotations

from datetime import timedelta

from livekit.api import AccessToken, VideoGrants


DEFAULT_TTL_HOURS = 2.0
DEFAULT_AGENT_IDENTITY = "saa-agent"


def attention_agent_token(
    *,
    api_key: str,
    api_secret: str,
    room_name: str,
    identity: str = DEFAULT_AGENT_IDENTITY,
    ttl_hours: float = DEFAULT_TTL_HOURS,
) -> str:
    """Issue a JWT for saa's hidden participant.

    Grants are minimum-necessary: `room_join`, `hidden`, `can_subscribe`,
    `can_publish_data`, `agent`. Never grants `can_publish` — the hosted
    agent never publishes media tracks, only data.

    Args:
        api_key:     Customer's LiveKit API key.
        api_secret:  Customer's LiveKit API secret.
        room_name:   Room the agent should join. Token is scoped to this room.
        identity:    Participant identity for the agent. Surfaced as
                     `session_handle.agent_identity` in the POST response so
                     the customer's voice agent can route upstream actions.
        ttl_hours:   Token validity window. Must outlast the intended session;
                     LK enforces expiry on connect, not mid-session, so a
                     long TTL is generally fine. Default 2 h.
    """
    grants = VideoGrants(
        room_join=True,
        room=room_name,
        hidden=True,
        can_subscribe=True,
        can_publish=False,
        can_publish_data=True,
        agent=True,
    )
    return (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_ttl(timedelta(hours=ttl_hours))
        .to_jwt()
    )
