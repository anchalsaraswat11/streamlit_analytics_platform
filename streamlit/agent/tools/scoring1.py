from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import boto3
import json
import os
from agent.tools.lookup import lookup_order

# The Scoring Tool is what actually calls our SageMaker model. It takes an order ID, 
# prepares the data the model expects, calls the endpoint, and returns a predicted customer LTV 
# — how much that customer is expected to spend in the next 90 days — along with a risk tier and a 
# plain English explanation

# The exact 39 columns my SageMaker endpoint was trained on 
TRAINING_COLUMNS = [
    'num_items', 'num_distinct_products', 'order_total',
    'shipping_matches_billing', 'user_session_count_7d',
    'user_page_views_7d', 'user_cart_adds_7d', 'user_cart_removals_7d',
    'user_searches_7d', 'account_age_days', 'previous_completed_purchases',
    'previous_returns', 'previous_chargebacks', 'historical_avg_order_value',
    'user_return_rate', 'avg_days_between_purchases', 'days_since_last_purchase',
    'primary_category_beauty', 'primary_category_books', 'primary_category_clothing',
    'primary_category_electronics', 'primary_category_food', 'primary_category_home',
    'primary_category_sports', 'primary_category_toys',
    'payment_method_apple_pay', 'payment_method_bank_transfer',
    'payment_method_credit_card', 'payment_method_debit_card',
    'payment_method_google_pay', 'payment_method_paypal',
    'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_USD',
    'primary_device_desktop', 'primary_device_mobile', 'primary_device_tablet'
]

# Columns to drop before encoding — not features
DROP_COLUMNS = ['order_id', 'user_id', 'customer_ltv_90d', 'churned_within_60d', 'returned_order']

SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def encode_row(raw_row: dict) -> pd.DataFrame:
    """
    Takes a raw feature row from lookup_order and produces
    the exact 39-column float vector the endpoint expects.
    """
    df = pd.DataFrame([raw_row])
    
    # Drop non-feature columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    
    # One-hot encode categoricals
    df = pd.get_dummies(df).astype(float)
    
    # Reindex to exact training column order, fill missing with 0
    df = df.reindex(columns=TRAINING_COLUMNS, fill_value=0.0)
    
    return df


def get_risk_tier(predicted_ltv: float) -> str:
    if predicted_ltv < 50:
        return "low"
    elif predicted_ltv < 150:
        return "medium"
    else:
        return "high"


def score_order(order_id: str) -> dict:
    """
    Looks up an order, encodes it, calls SageMaker, and returns
    predicted LTV, risk tier, and explanation.
    Never returns a score if the endpoint wasn't actually called.
    """
    # Step 1: lookup
    raw_row = lookup_order(order_id)
    if "error" in raw_row:
        return raw_row

    # Step 2: encode
    try:
        encoded = encode_row(raw_row)
    except Exception as e:
        return {"error": f"Encoding failed: {str(e)}"}

    # Step 3: call SageMaker
    try:
        client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        payload = encoded.to_csv(index=False, header=False)
        response = client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="text/csv",
            Body=payload
        )
        predicted_ltv = float(response['Body'].read().decode('utf-8').strip())
    except Exception as e:
        return {"error": f"SageMaker endpoint call failed: {str(e)}"}

    # Step 4: build result
    tier = get_risk_tier(predicted_ltv)
    result = {
        "order_id": order_id,
        "user_id": raw_row.get("user_id"),
        "predicted_ltv": round(predicted_ltv, 2),
        "risk_tier": tier,
        "explanation": (
            f"This customer is predicted to spend ${predicted_ltv:.2f} "
            f"in the next 90 days, placing them in the {tier} value tier."
        ),
        "top_factors": [
            f"avg_days_between_purchases: {raw_row.get('avg_days_between_purchases')}",
            f"previous_completed_purchases: {raw_row.get('previous_completed_purchases')}",
            f"user_return_rate: {raw_row.get('user_return_rate')}",
            f"user_cart_removals_7d: {raw_row.get('user_cart_removals_7d')}",
            f"historical_avg_order_value: {raw_row.get('historical_avg_order_value')}"
        ],
        "suggested_next_step": (
            "Priority follow-up recommended" if tier == "high"
            else "Standard handling" if tier == "medium"
            else "Monitor for churn risk"
        )
    }
    return json.dumps(result)


# Test encoding only — no SageMaker needed yet
if __name__ == "__main__":
    raw = lookup_order("ORD-17633387")
    encoded = encode_row(raw)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Columns match training: {list(encoded.columns) == TRAINING_COLUMNS}")
    print(f"Any nulls: {encoded.isnull().any().any()}")
    print()
    print("First few values:")
    print(encoded.iloc[0].to_dict())