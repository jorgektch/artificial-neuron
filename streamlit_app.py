import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Función de activación (sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definición de la neurona para el problema OR
def neuron_or(x1, x2):
    # Pesos y sesgo
    weights = np.array([1, 1])  # Pesos para cada entrada
    bias = -0.5  # Sesgo

    # Entrada ponderada
    z = np.dot(weights, [x1, x2]) + bias
    # Aplicación de la función de activación
    return sigmoid(z)

# Definición de la neurona para el problema XOR (2 capas)
def neuron_xor(x1, x2):
    # Capa oculta
    weights_hidden = np.array([[1, 1], [1, 1]])  # Pesos para la capa oculta
    bias_hidden = np.array([-0.5, -1.5])  # Sesgo para la capa oculta

    # Capa de salida
    weights_output = np.array([1, -2])  # Pesos para la capa de salida
    bias_output = 0.5  # Sesgo para la salida

    # Entrada ponderada de la capa oculta
    z_hidden = sigmoid(np.dot(weights_hidden, [x1, x2]) + bias_hidden)
    # Entrada ponderada de la salida
    z_output = np.dot(weights_output, z_hidden) + bias_output
    return sigmoid(z_output)

# Visualización de la estructura de la neurona
def plot_neuron_structure(neuron_type):
    if neuron_type == 'OR':
        st.write("### Estructura de la neurona para OR")
        st.image("neuron_or.png", caption="Estructura de la neurona OR")
    elif neuron_type == 'XOR':
        st.write("### Estructura de la neurona para XOR")
        st.image("neuron_xor.png", caption="Estructura de la neurona XOR")

# Aplicación en Streamlit
def main():
    st.title("Neurona para resolver los problemas OR y XOR")

    st.write("""
        En este proyecto, se presentan dos redes neuronales simples: una para resolver el problema OR y otra para resolver el problema XOR.
        Las neuronas tienen pesos y sesgos que se ajustan para generar las salidas correctas.
    """)

    st.sidebar.title("Selecciona un problema")
    problem_type = st.sidebar.selectbox("Problema", ['OR', 'XOR'])

    # Mostrar la estructura de la neurona seleccionada
    plot_neuron_structure(problem_type)

    # Entradas del usuario
    x1 = st.sidebar.slider("Entrada 1 (x1)", 0.0, 1.0, step=1.0)
    x2 = st.sidebar.slider("Entrada 2 (x2)", 0.0, 1.0, step=1.0)

    if problem_type == 'OR':
        output = neuron_or(x1, x2)
    elif problem_type == 'XOR':
        output = neuron_xor(x1, x2)

    st.write(f"### Resultado para el problema {problem_type}: {output:.4f}")

if __name__ == "__main__":
    main()
