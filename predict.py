
import pickle
import pandas as pd

model = pickle.load(
    open("src/model.pkl", "rb")
)

sample = pd.DataFrame({
    "area":[7420],
    "bedrooms":[4],
    "bathrooms":[2],
    "stories":[3],
    "mainroad":[1],
    "guestroom":[0],
    "basement":[1],
    "hotwaterheating":[0],
    "airconditioning":[1],
    "parking":[1],
    "prefarea":[1],
    "furnishingstatus":[1]
})

prediction = model.predict(sample)

print("Predicted Price:", prediction[0])
