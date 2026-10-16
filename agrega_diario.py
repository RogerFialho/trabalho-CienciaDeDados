"""Gera uma base DIÁRIA (1 linha por dia) a partir dos CSVs brutos 2009..2018.

Por que não usar o voos_limpos_2009_2018.parquet?
A remoção de outliers por IQR cortou todos os atrasos acima de 25 min e
zerou WEATHER_DELAY -- justamente os picos que a hipótese de sazonalidade
quer estudar. Aqui só fazemos a limpeza mínima (duplicatas) e agregamos.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
ANOS = range(2009, 2019)
CHAVE = ["FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME"]
COLS = CHAVE + ["DEP_DELAY", "CANCELLED", "CANCELLATION_CODE", "DIVERTED"]
SAIDA = DATA_DIR / "voos_diario_2009_2018.parquet"

diarios = []
for ano in ANOS:
    print(f"Lendo {ano}.csv ...")
    df = pd.read_csv(DATA_DIR / f"{ano}.csv", usecols=COLS,
                     dtype={"DEP_DELAY": "float32", "CANCELLED": "float32",
                            "DIVERTED": "float32", "CANCELLATION_CODE": "category"})
    # 2009.csv tem setembro duplicado (linhas idênticas) -> remove duplicatas
    antes = len(df)
    df = df.drop_duplicates(subset=CHAVE)
    if len(df) < antes:
        print(f"  {antes - len(df):,} linhas duplicadas removidas")
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    operou = df["CANCELLED"] == 0
    df["ATRASO15"] = (df["DEP_DELAY"] >= 15).where(operou)          # definição oficial de "voo atrasado"
    df["CANCEL_CLIMA"] = (df["CANCELLATION_CODE"] == "B").astype("float32")  # B = clima
    df["DEP_DELAY_OP"] = df["DEP_DELAY"].where(operou)

    d = df.groupby("FL_DATE").agg(
        n_voos=("FL_DATE", "size"),
        atraso_medio=("DEP_DELAY_OP", "mean"),
        atraso_mediano=("DEP_DELAY_OP", "median"),
        pct_atraso15=("ATRASO15", "mean"),
        taxa_cancel=("CANCELLED", "mean"),
        taxa_cancel_clima=("CANCEL_CLIMA", "mean"),
    )
    diarios.append(d)

diario = pd.concat(diarios).sort_index()
for c in ["pct_atraso15", "taxa_cancel", "taxa_cancel_clima"]:
    diario[c] *= 100   # em %
diario.to_parquet(SAIDA)
print(f"{len(diario):,} dias salvos em {SAIDA.name}")
print(diario.describe().round(2))
