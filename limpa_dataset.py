
import sys
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent

ANOS = range(2009, 2019)  # 2009..2018
ARQUIVOS = [DATA_DIR / f"{ano}.csv" for ano in ANOS]

COLUNAS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEP_DELAY",
    "TAXI_OUT",
    "WEATHER_DELAY",
    "CANCELLED",
    "DIVERTED",
]

DTYPES_LEITURA = {
    "OP_CARRIER": "category",
    "ORIGIN": "category",
    "DEP_DELAY": "float32",
    "TAXI_OUT": "float32",
    "WEATHER_DELAY": "float32",
    "CANCELLED": "float32",
    "DIVERTED": "float32",
}

ARQUIVO_SAIDA_PARQUET = DATA_DIR / "voos_limpos_2009_2018.parquet"
ARQUIVO_SAIDA_CSV = DATA_DIR / "voos_limpos_2009_2018.csv"



def ler_e_concatenar(arquivos: list[Path]) -> pd.DataFrame:
    dataframes = []

    for caminho in arquivos:
        if not caminho.exists():
            print(f"[AVISO] Arquivo não encontrado, pulando: {caminho.name}")
            continue

        try:
            print(f"Lendo {caminho.name} ...")
            df_ano = pd.read_csv(
                caminho,
                usecols=COLUNAS,
                dtype=DTYPES_LEITURA,
                low_memory=False,
            )
        except pd.errors.EmptyDataError:
            print(f"[AVISO] Arquivo vazio, pulando: {caminho.name}")
            continue
        except ValueError as erro:
            print(f"[ERRO] Falha ao ler {caminho.name}: {erro}")
            continue

        dataframes.append(df_ano)

    if not dataframes:
        raise RuntimeError(
            "Nenhum arquivo CSV pôde ser lido. Verifique se os arquivos "
            "2009.csv..2018.csv estão no diretório correto."
        )

    return pd.concat(dataframes, ignore_index=True)


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    # --- 3.1 Datas -----------------------------------------------------
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%Y-%m-%d", errors="coerce")

    # --- 3.2 Categorias (OP_CARRIER e ORIGIN já vieram como category da
    # leitura via dtype, mas garantimos aqui caso o df tenha sido montado
    # de outra forma / colunas não tenham herdado o dtype no concat) ----
    df["OP_CARRIER"] = df["OP_CARRIER"].astype("category")
    df["ORIGIN"] = df["ORIGIN"].astype("category")

    # --- 3.3 Nulos em WEATHER_DELAY: ausência de valor = sem atraso
    # climático registrado -> preenche com 0. ---------------------------
    df["WEATHER_DELAY"] = df["WEATHER_DELAY"].fillna(0)

    # --- 3.4 Nulos em DEP_DELAY / TAXI_OUT ------------------------------
    # Voos cancelados ou desviados legitimamente não têm esses tempos
    # registrados (o voo nunca decolou normalmente), então não devem ser
    # descartados por isso. Só removemos a linha quando o voo NÃO foi
    # cancelado nem desviado e, ainda assim, veio sem DEP_DELAY/TAXI_OUT
    # -- nesse caso é dado faltante de verdade, não uma ausência esperada.
    voo_normal = (df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)
    falta_atraso = df["DEP_DELAY"].isna() | df["TAXI_OUT"].isna()
    linhas_para_remover = voo_normal & falta_atraso

    if linhas_para_remover.any():
        print(
            f"Removendo {linhas_para_remover.sum():,} linhas de voos normais "
            "com DEP_DELAY/TAXI_OUT nulos."
        )
    df = df.loc[~linhas_para_remover].copy()

    # --- 3.5 Tipagem booleana -------------------------------------------
    # CANCELLED/DIVERTED chegam como float (0.0/1.0); convertê-las direto
    # para bool sem passar por Int seria arriscado se ainda houver NaN,
    # então garantimos que não sobrou nenhum antes do cast.
    for col in ("CANCELLED", "DIVERTED"):
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
        df[col] = df[col].astype(bool)

    # --- 3.6 Remoção de Outliers (Método dos Quartis / IQR) -------------
    # Aplica-se apenas às colunas numéricas de tempo/atraso
    colunas_numericas = ["DEP_DELAY", "TAXI_OUT", "WEATHER_DELAY"]
    
    for col in colunas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Mantemos apenas as linhas que estão dentro dos limites.
        # A condição `df[col].isna()` garante que não perderemos as linhas
        # de voos cancelados/desviados cujos NaNs foram preservados na etapa 3.4.
        mascara = ((df[col] >= limite_inferior) & (df[col] <= limite_superior)) | df[col].isna()
        linhas_outliers = (~mascara).sum()
        
        if linhas_outliers > 0:
            print(f"Removendo {linhas_outliers:,} outliers da coluna {col}.")
            
        df = df.loc[mascara].copy()

    return df


def exportar(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(ARQUIVO_SAIDA_PARQUET, index=False, compression="snappy")
        print(f"Arquivo salvo em Parquet: {ARQUIVO_SAIDA_PARQUET}")
    except ImportError:
        # Sem pyarrow/fastparquet instalado -> cai para CSV.
        print(
            "[AVISO] Nenhuma engine de Parquet disponível (instale "
            "'pyarrow'). Exportando para CSV como alternativa."
        )
        df.to_csv(ARQUIVO_SAIDA_CSV, index=False)
        print(f"Arquivo salvo em CSV: {ARQUIVO_SAIDA_CSV}")


def main() -> None:
    df_bruto = ler_e_concatenar(ARQUIVOS)
    print(f"Total de linhas lidas: {len(df_bruto):,}")

    df_limpo = limpar_dados(df_bruto)
    print(f"Total de linhas após limpeza: {len(df_limpo):,}")
    print(f"Uso de memória: {df_limpo.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    exportar(df_limpo)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as erro:
        print(f"[ERRO FATAL] {erro}")
        sys.exit(1)
