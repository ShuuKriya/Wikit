import pandas as pd
import random
import os

# ------------------------------
# 7 CATEGORIES + EXAMPLE MERCHANTS
# ------------------------------

CATEGORIES = {
    "Dining": [
        "ZOMATO ORDER", "SWIGGY PAYMENT", "STARBUCKS COFFEE",
        "KFC ONLINE", "MCDONALDS BILL", "DOMINOS PIZZA"
    ],
    "Shopping": [
        "AMAZON PAY", "FLIPKART ORDER", "MYNTRA ONLINE",
        "AJIO PURCHASE", "NYKAA PAYMENT", "CROMA STORE"
    ],
    "Fuel": [
        "HPCL PETROL PUMP", "BPCL FUEL", "SHELL GAS STATION",
        "INDIANOIL PUMP", "RELIANCE FUEL", "ESSAR PETRO"
    ],
    "Groceries": [
        "BIGBASKET ORDER", "JIOMART PURCHASE", "DMART PAYMENT",
        "NATURES BASKET", "SPENCERS RETAIL", "FRESHTOHOME"
    ],
    "Bills": [
        "AIRTEL POSTPAID", "JIO RECHARGE", "BESCOM ELECTRICITY",
        "TATA PLAY DTH", "ACT BROADBAND", "BSNL PAYMENT"
    ],
    "Entertainment": [
        "NETFLIX SUBSCRIPTION", "SPOTIFY PREMIUM", "HOTSTAR PLAN",
        "AMAZON PRIME VIDEO", "BOOKMYSHOW MOVIE", "GAANA PREMIUM"
    ],
    "Travel": [
        "UBER RIDE", "OLA TRIP", "IRCTC TICKET", "MAKEMYTRIP BOOKING",
        "REDBUS PAYMENT", "GOIBIBO HOTEL"
    ]
}

# ------------------------------
# Extra noise patterns to simulate real transactions
# ------------------------------

NOISE_PATTERNS = [
    "*ORDER #", " TXN ID ", " REF ", " UPI ", " @OKICICI ",
    " 1299.00 INR", " 450.00", " PAYMENT", " *ONLINE", "#88324",
    " - MUMBAI", " - BLR", " 99.00 Rs", " UPI@YBL"
]


def generate_sample(merchant):
    """Generate a noisy transaction string."""
    transaction = merchant

    # randomly add noise
    if random.random() < 0.7:
        transaction += random.choice(NOISE_PATTERNS)

    if random.random() < 0.5:
        transaction = random.choice(NOISE_PATTERNS) + " " + transaction

    return transaction.strip()


def generate_dataset(n_per_class=30):
    rows = []

    for category, merchants in CATEGORIES.items():
        for _ in range(n_per_class):
            merchant = random.choice(merchants)
            text = generate_sample(merchant)
            rows.append([text, category])

    df = pd.DataFrame(rows, columns=["transaction", "category"])
    return df.sample(frac=1).reset_index(drop=True)  # shuffle


def main():
    train_df = generate_dataset(n_per_class=30)
    test_df = generate_dataset(n_per_class=12)

    # Always save to the same directory where this script is located
    base = os.path.dirname(os.path.abspath(__file__))

    train_df.to_csv(os.path.join(base, "train.csv"), index=False)
    test_df.to_csv(os.path.join(base, "test.csv"), index=False)

    print("Dataset generated successfully!")
    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))


if __name__ == "__main__":
    main()
