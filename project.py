from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random

app = Flask(__name__)

data = pd.read_csv('base1.csv').dropna()

X = data.drop(columns=["game_name", "game_image", "set_number"])
y = data["set_number"]

model = DecisionTreeClassifier()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggested', methods=['POST'])
def suggested():
    age = int(request.form['age'])
    gender = int(request.form['gender'])

    input_data = [[age, gender]]
    WhatGame = model.predict(input_data)[0]

    SetGames = data[data['set_number'] == WhatGame][['game_name', 'game_image']].values.tolist()
    SetGames = SetGames[:3] 

    RelatedGames = data[data['set_number'] != WhatGame][['game_name', 'game_image']].sample(n=2).values.tolist()
    
    img = [[game[0], f"images/{game[1]}"] for game in SetGames]
    img2 = [[game[0], f"images/{game[1]}"] for game in RelatedGames]

    return render_template('result.html', WhatGame=WhatGame, SetGames=img, RelatedGames=img2)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
