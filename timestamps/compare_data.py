import json

with open('model_weights_preds.json', 'r') as file:
    pure_data = json.load(file)
with open('model_weights_scaler_preds.json', 'r') as file:
    scaled_data = json.load(file)
with open('model_weights_weekends_preds.json', 'r') as file:
    pure_weekend_data = json.load(file)
with open('model_weights_preds.json', 'r') as file:
    scaled_weekend_data = json.load(file)

pure = abs(sum(pure_data) / len(pure_data))
scaled = abs(sum(scaled_data) / len(scaled_data))
pure_weekend = abs(sum(pure_weekend_data) / len(pure_weekend_data))
scaled_weekend = abs(sum(scaled_weekend_data) / len(scaled_weekend_data))

mini = min(pure, scaled, pure_weekend, scaled_weekend)
if mini == pure:
    print("pure model")
if mini == scaled:
    print("scaled model")
if mini == pure_weekend:
    print("pure weekend model")
if mini == scaled_weekend:
    print("scaled weekend model")

print(pure, scaled, pure_weekend, scaled_weekend)




