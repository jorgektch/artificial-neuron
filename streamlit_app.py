import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Funciones de activación
def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Diccionario de funciones de activación
activation_functions = {
    'Escalón': step_function,
    'Sigmoide': sigmoid,
    'ReLU': relu,
    'Tangente Hiperbólica': tanh,
    'Softmax': softmax
}

# Definición de la neurona para el problema OR
def neuron_or(x1, x2, activation_fn):
    weights = np.array([1, 1])
    bias = -0.5
    z = np.dot(weights, [x1, x2]) + bias
    return activation_fn(z)

# Definición de la neurona para el problema XOR (2 capas)
def neuron_xor(x1, x2, activation_fn):
    weights_hidden = np.array([[1, 1], [1, 1]])
    bias_hidden = np.array([-0.5, -1.5])
    weights_output = np.array([1, -2])
    bias_output = 0.5
    z_hidden = sigmoid(np.dot(weights_hidden, [x1, x2]) + bias_hidden)
    z_output = np.dot(weights_output, z_hidden) + bias_output
    return activation_fn(z_output)

# Visualización de las funciones de activación
def plot_activation_function(activation_fn_name):
    x = np.linspace(-10, 10, 100)
    if activation_fn_name == 'Softmax':
        y = activation_functions[activation_fn_name](np.array([x, x]))
    else:
        y = activation_functions[activation_fn_name](x)

    plt.plot(x, y)
    plt.grid(True)
    st.pyplot(plt.gcf())

# Visualización de la estructura de la neurona
def plot_neuron_structure(neuron_type):
    fig, ax = plt.subplots()

    if neuron_type == 'OR':
        ax.set_title("Estructura de la Neurona OR")
        ax.text(0.2, 0.5, 'x1', fontsize=12, ha='center')
        ax.text(0.2, 0.2, 'x2', fontsize=12, ha='center')
        ax.text(0.5, 0.35, 'Σ', fontsize=20, ha='center')
        ax.arrow(0.3, 0.5, 0.15, -0.05, head_width=0.02, head_length=0.02, fc='k', ec='k')
        ax.arrow(0.3, 0.2, 0.15, 0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
        ax.text(0.8, 0.35, 'Output', fontsize=12, ha='center')
        ax.arrow(0.6, 0.35, 0.15, 0, head_width=0.02, head_length=0.02, fc='k', ec='k')

    elif neuron_type == 'XOR':
        ax.set_title("Estructura de la Neurona XOR")
        ax.text(0.1, 0.7, 'x1', fontsize=12, ha='center')
        ax.text(0.1, 0.3, 'x2', fontsize=12, ha='center')
        ax.text(0.4, 0.6, 'Σ', fontsize=20, ha='center')
        ax.text(0.4, 0.4, 'Σ', fontsize=20, ha='center')
        ax.text(0.7, 0.5, 'Output', fontsize=12, ha='center')
        ax.arrow(0.2, 0.7, 0.15, -0.05, head_width=0.02, head_length=0.02, fc='k', ec='k')
        ax.arrow(0.2, 0.3, 0.15, 0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
        ax.arrow(0.5, 0.6, 0.15, -0.05, head_width=0.02, head_length=0.02, fc='k', ec='k')
        ax.arrow(0.5, 0.4, 0.15, 0.05, head_width=0.02, head_length=0.02, fc='k', ec='k')

    ax.axis('off')
    st.pyplot(fig)

# Mostrar el código de la neurona
def display_code(neuron_type):
    if neuron_type == 'OR':
        code_or = '''# Definición de la neurona para el problema OR
def neuron_or(x1, x2):
    weights = np.array([1, 1])  # Pesos
    bias = -0.5  # Sesgo
    z = np.dot(weights, [x1, x2]) + bias
    return sigmoid(z)  # Función de activación sigmoide
'''
        st.code(code_or, language='python')
        st.write("""
        ### Explicación:
        Esta neurona tiene dos entradas (x1 y x2) con pesos iguales (1). El sesgo es -0.5, lo que permite que la 
        salida sea 0 cuando ambas entradas son 0. Utiliza la función de activación sigmoide para obtener la 
        salida entre 0 y 1, resolviendo el problema OR.
        """)

    elif neuron_type == 'XOR':
        code_xor = '''# Definición de la neurona para el problema XOR (2 capas)
def neuron_xor(x1, x2):
    weights_hidden = np.array([[1, 1], [1, 1]])  # Pesos capa oculta
    bias_hidden = np.array([-0.5, -1.5])  # Sesgo capa oculta
    weights_output = np.array([1, -2])  # Pesos capa de salida
    bias_output = 0.5  # Sesgo salida
    z_hidden = sigmoid(np.dot(weights_hidden, [x1, x2]) + bias_hidden)
    z_output = np.dot(weights_output, z_hidden) + bias_output
    return sigmoid(z_output)
'''
        st.code(code_xor, language='python')
        st.write("""
        ### Explicación:
        La neurona XOR utiliza dos capas. La primera capa (oculta) tiene dos neuronas que aplican una combinación
        de pesos y sesgos a las entradas. La segunda capa (salida) toma las salidas de la capa oculta y genera 
        el valor final. Esta estructura permite resolver el problema XOR, que no es linealmente separable.
        """)

# Interpretación del resultado
def interpret_result(problem_type, result):
    if problem_type == 'OR':
        if result >= 0.5:
            return "El resultado es 1: La salida es verdadera para el problema OR."
        else:
            return "El resultado es 0: La salida es falsa para el problema OR."
    elif problem_type == 'XOR':
        if result >= 0.5:
            return "El resultado es 1: La salida es verdadera para el problema XOR."
        else:
            return "El resultado es 0: La salida es falsa para el problema XOR."

# Aplicación en Streamlit
def main():
    st.title("Neurona para resolver los problemas OR y XOR")

    st.write("""
        En este proyecto, se presentan dos redes neuronales simples: una para resolver el problema OR y otra para resolver el problema XOR.
        Las neuronas tienen pesos y sesgos que se ajustan para generar las salidas correctas.
    """)

    st.sidebar.title("Configuración")
    problem_type = st.sidebar.selectbox("Selecciona un problema", ['OR', 'XOR'])
    activation_fn_name = st.sidebar.selectbox("Selecciona la función de activación", ['Escalón', 'Sigmoide', 'ReLU', 'Tangente Hiperbólica', 'Softmax'])

    activation_fn = activation_functions[activation_fn_name]

    # Mostrar la estructura de la neurona seleccionada
    plot_neuron_structure(problem_type)

    # Mostrar el código de la neurona seleccionada
    display_code(problem_type)

    # Mostrar la función de activación seleccionada
    st.write(f"### Función de activación seleccionada: {activation_fn_name}")
    plot_activation_function(activation_fn_name)

    # Entradas del usuario
    x1 = st.sidebar.slider("Entrada 1 (x1)", 0.0, 1.0, step=1.0)
    x2 = st.sidebar.slider("Entrada 2 (x2)", 0.0, 1.0, step=1.0)

    # Calcular la salida según el problema
    if problem_type == 'OR':
        output = neuron_or(x1, x2, activation_fn)
    elif problem_type == 'XOR':
        output = neuron_xor(x1, x2, activation_fn)

    st.write(f"### Resultado para el problema {problem_type}: {output:.4f}")

    # Mostrar la interpretación del resultado
    st.write("### Interpretación del resultado:")
    st.write(interpret_result(problem_type, output))

if __name__ == "__main__":
    main()
