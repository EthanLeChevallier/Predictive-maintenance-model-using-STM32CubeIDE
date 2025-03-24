import serial
import numpy as np
from sklearn.preprocessing import StandardScaler

PORT = "COM4"

# Fonction pour générer des données aléatoires (en tenant compte des mêmes caractéristiques)
def generate_random_data(n_samples):
    """
    Génère des données aléatoires pour les entrées et les étiquettes.
    Les valeurs des caractéristiques sont générées dans les mêmes plages que celles de notre dataset.
    """
    # Générer des entrées aléatoires (caractéristiques des machines)
    air_temperature = np.random.uniform(300, 350, n_samples)  # Température de l'air [K]
    process_temperature = np.random.uniform(350, 450, n_samples)  # Température du processus [K]
    rotational_speed = np.random.uniform(500, 5000, n_samples)  # Vitesse de rotation [rpm]
    torque = np.random.uniform(10, 100, n_samples)  # Couple [Nm]
    tool_wear = np.random.uniform(1, 150, n_samples)  # Usure de l'outil [min]

    # Empiler les caractéristiques dans un tableau numpy
    X = np.vstack((air_temperature, process_temperature, rotational_speed, torque, tool_wear)).T

    return X

# Fonction pour synchroniser la communication UART
def synchronise_UART(serial_port):
    """
    Synchronise la communication UART en envoyant un byte et en attendant une réponse.
    """
    print("Synchronisation en cours...")
    while True:
        serial_port.write(b"\xAB") 
        ret = serial_port.read(1)  
        if ret == b"\xCD":
            serial_port.read(1)  
            print("Synchronisation réussie.")
            break

# Fonction pour envoyer les entrées au STM32
def send_inputs_to_STM32(inputs, serial_port):
    inputs = inputs.astype(np.float32)
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()  
    serial_port.write(buffer)  

# Fonction pour lire les sorties du STM32
def read_output_from_STM32(serial_port):
    """
    Lit 6 bytes (pour les 6 classes de sortie) et les convertit en float.
    """
    output = serial_port.read(6)  
    print(f"Sortie brute lue : {output}")  

    # Vérifier si la sortie n'est pas vide
    if len(output) == 0:
        print("Aucune donnée reçue depuis STM32.")
        return [0.0] * 6  

    float_values = [int(out) / 255 for out in output]  
    return float_values

# Fonction pour évaluer le modèle sur STM32
def evaluate_model_on_STM32(X_test, y_true, serial_port, iterations=100):
    accuracy = 0
    for i in range(iterations):
        send_inputs_to_STM32(X_test[i], serial_port)
        output = read_output_from_STM32(serial_port)

        if len(output) == 0:
            continue

        pred_class = np.argmax(output)
        true_class = y_true[i]

        if pred_class == true_class:
            accuracy += 1 / iterations

        print(f"STM32 → {pred_class}, attendu → {true_class}")

    return accuracy

# Programme principal
def main():
    # Générer des données de test aléatoires
    n_samples = 100  
    X_test = generate_random_data(n_samples)

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test) 

    # Ouverture du port série pour communiquer avec STM32
    try:
        with serial.Serial(PORT, 115200, timeout=1) as ser:
            print("Synchronisation avec le STM32...")
            synchronise_UART(ser)  
            print("Synchronisé.")

            # Évaluation du modèle sur STM32
            print("Évaluation du modèle sur STM32...")
            y_test = np.random.randint(0, 6, size=n_samples)

            accuracy = evaluate_model_on_STM32(X_test_scaled, y_test, ser, iterations=100)
            print(f"Précision sur STM32 : {accuracy:.2f}")
    except Exception as e:
        print(f"Erreur de communication avec le STM32 : {e}")

if __name__ == '__main__':
    main()
