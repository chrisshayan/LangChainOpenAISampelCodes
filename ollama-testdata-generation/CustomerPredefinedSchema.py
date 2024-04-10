import requests
import json

model = "llama2"
template = {
    "firstName": "",
    "lastName": "",
    "address": {
        "street": "",
        "city": "",
        "country": "",
        "zipCode": ""
    },
    "phoneNumber": "",
    "account": {
        "account_number": "",
        "balance": "",
        "transactions": [
            {
                "dateTimeHHMMSS": "",
                "description": "",
                "amount": ""
            }
        ]
    },
    "creditCardNumber": "",
    "creditCardLimit": "",
    "debitCardNumber": ""
}

prompt = (f"generate one realistically believable sample data set of a persons first name, last name, address in the "
          f"Vietnam, phone number, current account number, balance for current account in VND, transaction detail (at least 10 records) "
          f"including date and time, description (deposit, withdraw or interest), realistic VISA or"
          f"Mastercard credit card"
          f"number, limit for credit card using VND, and a VISA or Mastercard debit card number. \nUse the following "
          f"template:"
          f"{json.dumps(template)}.")

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}

print(f"Generating a sample user")
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)
print(json.dumps(json.loads(json_data["response"]), indent=2))

# ollama pull llama2
# python3 CustomerPredefinedSchema.py
