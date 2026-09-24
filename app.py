"""
Análise de Crédito - Financiamento de Motos
=============================================
Aplicação Streamlit com modelo de classificação em TensorFlow
para simular a análise de aprovação de crédito de um banco.

Stack: Streamlit + TensorFlow
Deploy: Render (Web Service)
"""

import streamlit as st
import numpy as np
import tensorflow as tf

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Financia+ | Análise de Crédito",
    page_icon="🏍️",
    layout="centered",
)

# --- Identidade visual "banco sério" ---------------------------------------
st.markdown(
    """
    <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        [data-testid="stMain"], .main .block-container {
            background-color: #0B1C2C;
        }
        [data-testid="stHeader"] {
            background-color: transparent;
        }
        h1, h2, h3, p, label, span,
        [data-testid="stMarkdownContainer"] p {
            color: #EAF0F6;
        }
        .banco-header {
            background: linear-gradient(90deg, #0B1C2C 0%, #12314A 100%);
            padding: 1.2rem 1.5rem;
            border-radius: 10px;
            border: 1px solid #1F4B6E;
            margin-bottom: 1.5rem;
        }
        .banco-header h1 {
            margin: 0;
            font-size: 1.6rem;
            letter-spacing: 0.5px;
        }
        .banco-header p {
            margin: 0.2rem 0 0 0;
            color: #9FB6C9;
            font-size: 0.9rem;
        }
        .resultado-caixa {
            padding: 1.3rem 1.5rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .aprovado {
            background-color: #123B27;
            border: 1px solid #2E8B57;
        }
        .negado {
            background-color: #3B1414;
            border: 1px solid #B0413E;
        }
        .footer-disclaimer {
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #1F4B6E;
            color: #6E869C;
            font-size: 0.75rem;
        }
        div[data-testid="stForm"] {
            background-color: #12212F;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #1F4B6E;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="banco-header">
        <h1>🏍️ Financia+ — Crédito para sua primeira moto</h1>
        <p>Simulação de análise de crédito automatizada</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 1. DADOS SINTÉTICOS DE TREINAMENTO
# =========================================================
# Regras que o modelo aprende a partir dos dados:
#   - Idade mínima de 18 anos
#   - Comprometimento de renda com a parcela não deve ultrapassar ~30%
#   - Renda mínima de referência
# Uma pequena taxa de ruído é injetada para simular exceções reais
# de política de crédito (analista humano, histórico, etc.)
@st.cache_data
def gerar_dados_treino(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    idade = rng.integers(16, 75, n).astype(np.float32)
    renda = rng.uniform(900, 16000, n).astype(np.float32)
    parcela = rng.uniform(50, 3500, n).astype(np.float32)

    comprometimento = parcela / renda
    aprovado = (
        (idade >= 18)
        & (renda >= 1200)
        & (comprometimento <= 0.30)
    ).astype(np.float32)

    # Ruído controlado (~3%) para evitar um classificador
    # perfeitamente "regra-based" e simular exceções reais.
    flip_mask = rng.random(n) < 0.03
    aprovado = np.where(flip_mask, 1.0 - aprovado, aprovado)

    X = np.column_stack([idade, renda, parcela]).astype(np.float32)
    y = aprovado.astype(np.float32)
    return X, y


X_treino, y_treino = gerar_dados_treino()

# =========================================================
# 2. NORMALIZAÇÃO
# =========================================================
X_mean = X_treino.mean(axis=0)
X_std = X_treino.std(axis=0)


def normalizar(X):
    return (X - X_mean) / X_std


# =========================================================
# 3. TREINAMENTO DO MODELO (cacheado — roda uma única vez)
# =========================================================
@st.cache_resource
def treinar_modelo():
    X_norm = normalizar(X_treino)

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    modelo.fit(X_norm, y_treino, epochs=25, batch_size=64, verbose=0)
    return modelo


with st.spinner("Carregando motor de análise de crédito..."):
    modelo = treinar_modelo()


# =========================================================
# 4. FUNÇÕES DE PREDIÇÃO E SUGESTÃO
# =========================================================
def prever_aprovacao(idade: float, renda: float, parcela: float) -> float:
    """Retorna a probabilidade (0-1) de aprovação de crédito."""
    entrada = np.array([[idade, renda, parcela]], dtype=np.float32)
    entrada_norm = normalizar(entrada)
    prob = modelo(entrada_norm, training=False).numpy()[0][0]
    return float(prob)


def sugerir_parcela(idade: float, renda: float, parcela_atual: float, limiar: float = 0.5):
    """
    Busca binária pelo maior valor de parcela que o modelo aprovaria,
    mantendo idade e renda fixas. Retorna None se nenhuma parcela
    razoável (>= R$ 50) for aprovável.
    """
    baixo, alto = 50.0, parcela_atual
    if prever_aprovacao(idade, renda, baixo) < limiar:
        return None  # nem a menor parcela testada seria aprovada

    for _ in range(40):
        meio = (baixo + alto) / 2
        if prever_aprovacao(idade, renda, meio) >= limiar:
            baixo = meio
        else:
            alto = meio

    return round(baixo, 2)


# =========================================================
# 5. FORMULÁRIO
# =========================================================
st.subheader("📋 Dados para simulação")

with st.form("form_credito"):
    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input(
            "Idade", min_value=16, max_value=100, value=22, step=1
        )
    with col2:
        renda = st.number_input(
            "Renda mensal (R$)", min_value=0.0, max_value=100000.0,
            value=2200.0, step=100.0, format="%.2f",
        )

    parcela = st.number_input(
        "Valor da parcela pretendida (R$)", min_value=0.0, max_value=20000.0,
        value=450.0, step=50.0, format="%.2f",
    )

    enviado = st.form_submit_button("Analisar Crédito", use_container_width=True)

# =========================================================
# 6. RESULTADO
# =========================================================
if enviado:
    if idade < 18:
        st.markdown(
            '<div class="resultado-caixa negado">'
            '<h3>❌ Crédito não elegível</h3>'
            '<p>É necessário ter 18 anos ou mais para solicitar financiamento.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        probabilidade = prever_aprovacao(idade, renda, parcela)
        aprovado = probabilidade >= 0.5

        if aprovado:
            st.markdown(
                f"""
                <div class="resultado-caixa aprovado">
                    <h3>✅ Crédito aprovado</h3>
                    <p>Parabéns! Com base nos dados informados, seu financiamento
                    de <b>R$ {parcela:,.2f}</b>/mês foi pré-aprovado.</p>
                    <p style="color:#9FB6C9; font-size:0.85rem;">
                        Confiança do modelo: {probabilidade*100:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            sugestao = sugerir_parcela(idade, renda, parcela)

            if sugestao:
                st.markdown(
                    f"""
                    <div class="resultado-caixa negado">
                        <h3>❌ Crédito negado</h3>
                        <p>A parcela de <b>R$ {parcela:,.2f}</b>/mês compromete uma
                        parte muito alta da sua renda declarada.</p>
                        <p style="color:#9FB6C9; font-size:0.85rem;">
                            Confiança do modelo: {(1-probabilidade)*100:.1f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info(
                    f"💡 **Sugestão:** com uma parcela de até "
                    f"**R$ {sugestao:,.2f}/mês**, a simulação indica que seu "
                    f"crédito seria aprovado."
                )
            else:
                st.markdown(
                    """
                    <div class="resultado-caixa negado">
                        <h3>❌ Crédito negado</h3>
                        <p>Com a renda informada, não encontramos um valor de
                        parcela dentro dos parâmetros de aprovação. Recomendamos
                        procurar um de nossos consultores para avaliar outras
                        opções de financiamento.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown(
    """
    <div class="footer-disclaimer">
        Esta é uma simulação automatizada baseada em modelo estatístico e não
        constitui uma oferta de crédito. A aprovação final está sujeita à
        análise cadastral completa. Financia+ é uma aplicação de demonstração.
    </div>
    """,
    unsafe_allow_html=True,
)
