import os

# Full data source
CSV_DATA = os.getenv("CSV_DATA", None)
if not CSV_DATA:
    CSV_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv"))

NUMERICAL_FEATURES = [
    "account_amount_added_12_24m",
    "account_days_in_dc_12_24m",
    "account_days_in_rem_12_24m",
    "account_days_in_term_12_24m",
    "account_incoming_debt_vs_paid_0_24m",
    "age",
    "avg_payment_span_0_12m",
    "avg_payment_span_0_3m",
    "max_paid_inv_0_12m",
    "max_paid_inv_0_24m",
    "num_active_div_by_paid_inv_0_12m",
    "num_active_inv",
    "num_arch_dc_0_12m",
    "num_arch_dc_12_24m",
    "num_arch_ok_0_12m",
    "num_arch_ok_12_24m",
    "num_arch_rem_0_12m",
    "num_arch_written_off_0_12m",
    "num_arch_written_off_12_24m",
    "num_unpaid_bills",
    "recovery_debt",
    "sum_capital_paid_account_0_12m",
    "sum_capital_paid_account_12_24m",
    "sum_paid_inv_0_12m",
    "time_hours",
]

CATEGORICAL_FEATURES = [
    "account_status",
    "account_worst_status_0_3m",
    "account_worst_status_12_24m",
    "account_worst_status_3_6m",
    "account_worst_status_6_12m",
    "merchant_category",
    "merchant_group",
    "has_paid",
    "name_in_email",
    "status_last_archived_0_24m",
    "status_2nd_last_archived_0_24m",
    "status_3rd_last_archived_0_24m",
    "status_max_archived_0_6_months",
    "status_max_archived_0_12_months",
    "status_max_archived_0_24_months",
    "worst_status_active_inv",
]

ENCODED_FEATURES = [
    "merchant_category",
    "merchant_group",
    "name_in_email",
]

BINNED_FEATURES = [
    "time_hours",
    "age",
    "max_paid_inv_0_24m",
    "avg_payment_span_0_3m",
    "num_arch_ok_0_12m",
    "num_arch_ok_12_24m"
]
