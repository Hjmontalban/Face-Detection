import pandas as pd

# Define the label_names dictionary with person information
label_names = {
    0: {"name": "Humphrey John M. Montalban", "age": 23, "department": "CICT", "student_number": "2021M0651"},
    1: {"name": "Lima, Dave Emanuel, G", "age": 21, "department": "CICT", "student_number": "2021M0869"},
    2: {"name": "Gregorios, Christia Tiffany Manuelle Q.", "age": 22, "department": "CICT", "student_number": "2021M0010"},
    3: {"name": "Singh, Reeman L.", "age": 21, "department": "CICT", "student_number": "2021M0177"},
    4: {"name": "Pontillas, Steven Ken E.", "age": 19, "department": "CICT", "student_number": "2023M0282"},
    5: {"name": "Lamsin, Gloria Marie P.", "age": 22, "department": "CICT", "student_number": "2021M0059"},
    6: {"name": "Apinan, Fatima Grace, T.", "age": 23, "department": "CICT", "student_number": "2021M0005"},
    7: {"name": "Sarmiento, Reycel B.", "age": 21, "department": "CICT", "student_number": "2022M0252"},
    8: {"name": "Española, Lyann Marie S", "age": 20, "department": "COC", "student_number": "2022M0153"},
    9: {"name": "Barcelona, David Bryan S.", "age": 22, "department": "CICT", "student_number": "2021M0119"},
    10: {"name": "Juaneza, Jaymar A", "age": 22, "department": "CICT", "student_number": "2021M0066"},
    11: {"name": "Loot, Dana Jamela T.", "age": 19, "department": "COC", "student_number": "2023M0410"},
    12: {"name": "Gulmatico, Rod Ian H.", "age": 22, "department": "COC", "student_number": "2021M0899"},
    13: {"name": "Sabolbora Windy C.", "age": 21, "department": "CICT", "student_number": "2021M0073"},
    14: {"name": "Dan Axcel A. Arnigo", "age": 20, "department": "COC", "student_number": "2023M1277"},
    15: {"name": "Noblezada, Yestin Bonn T.", "age": 19, "department": "COC", "student_number": "2023M2070"},
    16: {"name": "Lebeco, Yhiennel C.", "age": 19, "department": "COC", "student_number": "2023M1284"},
    17: {"name": "Porras, Christine Elijah F.", "age": 19, "department": "COC", "student_number": "2023M1525"},
    18: {"name": "Sarah Nicole Calinagan", "age": 21, "department": "CICT", "student_number": "2021M0053"},
    19: {"name": "Angot, Ike, J.", "age": 23, "department": "CICT", "student_number": "2021M0011"},
    20: {"name": "Nillos Jasper M.", "age":22, "department": "CICT", "student_number": "2021M0121"},
    21: {"name": "De Leon Kyne.", "age": 22, "department": "CICT", "student_number": "2021M0051"},
    22: {"name": "Suarnaba Maria April.", "age": 22, "department": "CICT", "student_number": "2021M0652"},
    23: {"name": "Jagunap Roan H.", "age": 22, "department": "CICT", "student_number": "2021M0212"}
}

# Convert the label_names dictionary into a list of dictionaries
data = [
    {"Label ID": label_id, **info}
    for label_id, info in label_names.items()
]

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
excel_file_path = "face_recognition_labels.xlsx"
try:
    import openpyxl
    df.to_excel(excel_file_path, index=False)
    print(f"Data saved to {excel_file_path}")
except ModuleNotFoundError:
    print("Error: 'openpyxl' module is not installed. Please install it by running 'pip install openpyxl'.")

# Save the DataFrame to a CSV file
csv_file_path = "face_recognition_labels.csv"
df.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")
