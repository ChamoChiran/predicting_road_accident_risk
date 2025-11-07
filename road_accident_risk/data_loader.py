import pandas as pd
from .config import EXPECTED_COLUMNS


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _pandas_nullable_dtype(expected_dtype: str) -> str:
    """
    Convert expected dtype string to pandas dtype string.

    Examples:
        - "int64" -> "int64"
        - "float64" -> "float64"
        - "bool" -> "bool"
        - "object" -> "object"
    """
    if expected_dtype.startswith("int"):
        return "int64"
    if expected_dtype.startswith("float"):
        return "float64"
    if expected_dtype == "bool":
        return "bool"
    return "object"



def load_user_input(user_dict: dict) -> pd.DataFrame:
    """
    Convert user input dictionary to a properly formatted DataFrame.

    This function:
    1. Validates that all required fields are present
    2. Converts values to the correct data types
    3. Returns a single-row DataFrame ready for model prediction

    Args:
        user_dict: Dictionary containing all required input fields

    Returns:
        Single-row DataFrame with properly typed columns

    Raises:
        ValueError: If any required fields are missing
    """

    # convert dict to dataframe
    df = pd.DataFrame([user_dict])

    # convert each column to correct data type (predicting ready)
    for col, expected_dtype in EXPECTED_COLUMNS.items():
        pandas_dtype = _pandas_nullable_dtype(expected_dtype)

        if pandas_dtype == "object":
            # Convert to string
            df[col] = df[col].astype(str)

        elif pandas_dtype == "float64":
            # Convert to float (handles string numbers like "0.5")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        elif pandas_dtype == "int64":
            # Convert to integer (handles string numbers like "2")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")

        elif pandas_dtype == "bool":
            # Convert to boolean
            if pd.api.types.is_bool_dtype(df[col].dtype):
                # Already a boolean
                df[col] = df[col].astype("bool")
            else:
                # Convert strings like "true", "false", "yes", "no" to boolean
                mapped = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False
                    })
                )
                df[col] = mapped.astype("bool")

    # reorder columns to match expected dataframe
    df = df[list(EXPECTED_COLUMNS.keys())]

    return df
