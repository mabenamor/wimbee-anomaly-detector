import numpy as np
import pandas as pd
import random
from datetime import timedelta

def generate_murex_data_less_extreme(
    n_samples=100_000,
    anomaly_ratio=0.01,
    random_seed=42
):
    """
    Génère un DataFrame contenant des transactions Murex-like,
    avec ~1% d’anomalies moins extrêmes pour un aspect plus réaliste.

    Colonnes:
    [TimeStamp, tradeDate, settlementDate, TraderID, TradeID, Action, Product,
     NotionalAmount, Counterparty, PriceOrRate, Status, Latency(ms), is_anomaly]
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1) Génération de plages de dates
    # -------------------------------------------------------------------------
    date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='H')
    timestamps = np.random.choice(date_range, size=n_samples, replace=True)

    # tradeDate = TimeStamp + offset en heures
    trade_date_offsets = np.random.randint(0, 24, size=n_samples)
    trade_dates = [
        ts + pd.Timedelta(hours=int(offset))
        for ts, offset in zip(timestamps, trade_date_offsets)
    ]

    # settlementDate = tradeDate + offset entre 1 et 30 jours
    settlement_offsets = np.random.randint(1, 30, size=n_samples)
    settlement_dates = [
        td + pd.Timedelta(days=int(offset))
        for td, offset in zip(trade_dates, settlement_offsets)
    ]

    # 2) Listes pour la génération aléatoire de champs
    # -------------------------------------------------------------------------
    trader_ids = [f"TRADER_{i}" for i in range(1, 201)]
    actions = ["BUY", "SELL"]
    products = ["Futures", "Options", "FXSpot", "Swap", "Bond"]
    counterparties = ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E"]
    statuses = ["Confirmed", "Pending", "Cancelled"]

    # 3) Génération aléatoire des valeurs "normales"
    # -------------------------------------------------------------------------
    # NotionalAmount ~ gaussien autour de 1 million
    notional_amount = np.random.normal(loc=1_000_000, scale=300_000, size=n_samples)
    notional_amount = np.clip(notional_amount, 50_000, 3_000_000)  # limiter la plage

    # PriceOrRate ~ gaussien autour de 1.2
    price_or_rate = np.random.normal(loc=1.2, scale=0.3, size=n_samples)
    price_or_rate = np.clip(price_or_rate, 0.3, 5.0)  # on évite des valeurs négatives ou trop extrêmes

    # Latency(ms) ~ gaussien autour de 100
    latency = np.random.normal(loc=100, scale=30, size=n_samples)
    latency = np.clip(latency, 1, 500)  # latence entre 1 ms et 500 ms max

    trader_id_series = np.random.choice(trader_ids, size=n_samples)
    action_series = np.random.choice(actions, size=n_samples)
    product_series = np.random.choice(products, size=n_samples)
    counterparty_series = np.random.choice(counterparties, size=n_samples)
    status_series = np.random.choice(statuses, size=n_samples)

    # 4) Construction du DataFrame
    # -------------------------------------------------------------------------
    df = pd.DataFrame({
        "TimeStamp": timestamps,
        "tradeDate": trade_dates,
        "settlementDate": settlement_dates,
        "TraderID": trader_id_series,
        "TradeID": [f"T_{i+1}" for i in range(n_samples)],
        "Action": action_series,
        "Product": product_series,
        "NotionalAmount": notional_amount,
        "Counterparty": counterparty_series,
        "PriceOrRate": price_or_rate,
        "Status": status_series,
        "Latency(ms)": latency
    })

    # 5) Insertion d'anomalies ~ 1%
    # -------------------------------------------------------------------------
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

    anomaly_types = [
        "settlement_before_trade",
        "mild_extreme_price",
        "mild_extreme_notional",
        "mild_abnormal_latency"
    ]
    # On retire l'option "invalid_trader_id" ou on peut la laisser si besoin

    for idx in anomaly_indices:
        anomaly_type = random.choice(anomaly_types)

        if anomaly_type == "settlement_before_trade":
            # offset aléatoire entre 1 et 5 jours plus tôt
            days_back = np.random.randint(1, 6)
            df.at[idx, "settlementDate"] = df.at[idx, "tradeDate"] - timedelta(days=days_back)

        elif anomaly_type == "mild_extreme_price":
            # Valeurs un peu décalées : ex. environ 0.1 à 0.5 ou 5 à 10
            # Au lieu de 0.0001 ou 1000
            if random.random() < 0.5:
                # léger
                df.at[idx, "PriceOrRate"] = round(np.random.uniform(0.1, 0.5), 3)
            else:
                # un peu élevé
                df.at[idx, "PriceOrRate"] = round(np.random.uniform(5, 8), 3)

        elif anomaly_type == "mild_extreme_notional":
            # Normalement ~ 1M, on peut sortir un peu : ex. 3M - 8M
            if random.random() < 0.5:
                df.at[idx, "NotionalAmount"] = round(np.random.uniform(5e5, 8e5), -3)  # un peu sous la moyenne
            else:
                df.at[idx, "NotionalAmount"] = round(np.random.uniform(2e6, 5e6), -3)  # un peu au-dessus

        elif anomaly_type == "mild_abnormal_latency":
            # latence négative ou trop haute, mais pas extrême
            # ex. -5 à -1 ms ou 600-800 ms
            if random.random() < 0.5:
                df.at[idx, "Latency(ms)"] = np.random.randint(-5, 0)  # légérement négatif
            else:
                df.at[idx, "Latency(ms)"] = np.random.randint(600, 800)  # trop haut, mais pas 999999

    # Ajouter la colonne is_anomaly
    df["is_anomaly"] = 0
    df.loc[anomaly_indices, "is_anomaly"] = 1

    return df

if __name__ == "__main__":
    print("Génération d'un dataset avec anomalies moins extrêmes...")
    df_data = generate_murex_data_less_extreme(
        n_samples=100_000,
        anomaly_ratio=0.01,
        random_seed=42
    )
    print(f"DataFrame généré : {df_data.shape[0]} lignes, {df_data.shape[1]} colonnes.")



    # Sauvegarde éventuelle en CSV
    df_data.to_csv("murex_data_less_extreme.csv", index=False)
    print("Fichier 'murex_data_less_extreme.csv' créé.")
