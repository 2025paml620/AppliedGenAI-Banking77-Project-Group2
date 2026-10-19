"""Decoding configurations for E3."""
DECODING = {
    "greedy": {
        "do_sample": False
    },
    "temperature_0_2": {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 1.0
    },
    "temperature_0_7": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 1.0
    },
    "top_p_0_9": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }
}
