# 📦 Template Prático: Regressão Logística para E-Commerce & Classificação Binária

Este repositório contém um template modular e pronto para produção em **Python (scikit-learn)** para problemas de classificação binária, aplicado ao cenário real de **Previsão de Devolução de Pedidos (E-Commerce)**.

---

## 🎯 Problema de Negócio
No setor de e-commerce (especialmente vestuário), o custo logístico de devoluções e trocas impacta diretamente a margem de lucro. O objetivo deste projeto é prever quais pedidos têm alta probabilidade de serem devolvidos *antes do envio*, permitindo ações preventivas (como alertas de caimento e validação de tamanho).

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `Pipeline`, `ColumnTransformer`)
* **Visualização:** Matplotlib, Seaborn

---

## 🚀 Principais Destaques do Template
1. **Prevenção de Vazamento de Dados (Data Leakage):** Remoção de variáveis pós-evento (ex: `return_reason`).
2. **Pipeline de Pré-Processamento Automático:**
   * Padronização de variáveis numéricas (`StandardScaler`).
   * Codificação de variáveis categóricas (`OneHotEncoder`).
3. **Ajuste de Limiar de Decisão (Threshold Tuning):**
   * Otimização do limite de decisão para **35%**, elevando o **Recall de Devoluções para 66%** e maximizando a captura de casos críticos para a operação.
4. **Interpretação para o Negócio (Odds Ratio):**
   * Extração dos coeficientes logísticos convertidos em Razão de Chances (Odds Ratio) para indicar ao time comercial quais categorias e atributos mais geram devolução.

---

## 📊 Principais Descobertas do Modelo
* **Alto Risco de Devolução:** Categorias de **Calças e Jeans** apresentaram *Odds Ratio* superior a **3.5x**, enquanto o descompasso de tamanho (`size_mismatch`) **quase triplica (2.9x)** a chance de retorno.
* **Fatores de Proteção:** Produtos com caimento fiel (*True to Size*) e tecidos com elastano reduziram significativamente a taxa de devolução.

---

## 💻 Como Executar
1. Clone o repositório:
   ```bash
   git clone [https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git](https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git)
   ```
2. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Execute o script ou abra o arquivo no Google Colab/Jupyter Notebook.
