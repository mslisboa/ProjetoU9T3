# ProjetoU9T3



Aqui está o plano detalhado com os blocos de código necessários para realizar a tarefa descrita. A variável dependente será a **"taxa de engajamento"** (por exemplo, `60_day_eng_rate`), e utilizaremos Python para desenvolver e otimizar o modelo de regressão linear com gradiente descendente, regularização e técnicas de validação cruzada.

---

### **1. Importação de Bibliotecas e Preparação do Dataset**

Primeiro, realizamos a importação das bibliotecas, o carregamento e a preparação do dataset.

```python
# Importação de bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Supondo que 'numeric_dataset' já contenha os dados:
# Remover linhas com valores nulos
numeric_dataset.dropna(inplace=True)

# Separar a variável dependente (taxa de engajamento) e as independentes
y = numeric_dataset["60_day_eng_rate"]
X = numeric_dataset.drop(columns=["60_day_eng_rate"])

# Dividir os dados em treino (60%) e teste (40%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
```

---

### **2. Normalização dos Dados**

Padronizamos as variáveis para melhorar a convergência durante o treinamento.

```python
# Normalização (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### **3. Implementação do Algoritmo de Regressão Linear**

Iniciaremos com um modelo de regressão linear simples utilizando mínimos quadrados.

```python
# Regressão Linear simples
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Previsões
y_pred = linear_model.predict(X_test_scaled)

# Avaliação do modelo
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
```

---

### **4. Otimização e Ajustes: Gradiente Descendente**

Implementamos o treinamento manual usando gradiente descendente.

```python
# Função de custo
def compute_cost(X, y, weights):
    n = len(y)
    predictions = X @ weights
    cost = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
    return cost

# Gradiente descendente
def gradient_descent(X, y, weights, learning_rate, epochs):
    n = len(y)
    cost_history = []

    for i in range(epochs):
        predictions = X @ weights
        gradients = (1 / n) * (X.T @ (predictions - y))
        weights -= learning_rate * gradients
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# Configuração
X_train_manual = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]  # Adicionar bias (1s)
X_test_manual = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]
weights = np.zeros(X_train_manual.shape[1])  # Inicializar pesos
learning_rate = 0.01
epochs = 1000

# Treinamento
final_weights, cost_history = gradient_descent(X_train_manual, y_train.values, weights, learning_rate, epochs)

# Previsões e avaliação
y_pred_gd = X_test_manual @ final_weights
r2_gd = r2_score(y_test, y_pred_gd)
mse_gd = mean_squared_error(y_test, y_pred_gd)
mae_gd = mean_absolute_error(y_test, y_pred_gd)

print(f"Gradiente Descendente -> R²: {r2_gd}, MSE: {mse_gd}, MAE: {mae_gd}")
```

---

### **5. Regularização: Lasso e Ridge**

Aplicamos regularização para reduzir o overfitting.

```python
# Ridge (L2)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Lasso (L1)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Avaliação Ridge
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print(f"Ridge -> R²: {r2_ridge}, MSE: {mse_ridge}, MAE: {mae_ridge}")

# Avaliação Lasso
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print(f"Lasso -> R²: {r2_lasso}, MSE: {mse_lasso}, MAE: {mae_lasso}")
```

---

### **6. Validação Cruzada**

Utilizamos validação cruzada para avaliar o modelo.

```python
from sklearn.model_selection import cross_val_score

# Validação cruzada para Ridge
cv_ridge = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring="r2")
print(f"Validação Cruzada (Ridge) -> R² médio: {cv_ridge.mean()}")

# Validação cruzada para Lasso
cv_lasso = cross_val_score(lasso_model, X_train_scaled, y_train, cv=5, scoring="r2")
print(f"Validação Cruzada (Lasso) -> R² médio: {cv_lasso.mean()}")
```

---

### **7. Visualização dos Resultados**

#### Gráfico de Convergência do Gradiente Descendente
```python
import matplotlib.pyplot as plt

plt.plot(range(epochs), cost_history)
plt.xlabel("Épocas")
plt.ylabel("Custo")
plt.title("Convergência do Gradiente Descendente")
plt.show()
```

#### Gráfico de Resíduos
```python
residuals = y_test - y_pred_ridge
plt.scatter(y_test, residuals)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Valores Reais")
plt.ylabel("Resíduos")
plt.title("Gráfico de Resíduos (Ridge)")
plt.show()
```

---

### **8. Conclusões**
- Compare as métricas de desempenho entre os diferentes modelos (Linear, Ridge, Lasso, e Gradiente Descendente).
- Interprete os coeficientes dos modelos regularizados para entender a relevância das variáveis.
- Escolha o modelo com melhor equilíbrio entre desempenho e simplicidade (geralmente com validação cruzada).

Se precisar de ajustes, posso ajudá-lo a refinar qualquer etapa!
