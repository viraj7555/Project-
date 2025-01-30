# from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
# import pandas as pd
# import joblib
# import sqlite3
# from werkzeug.security import generate_password_hash, check_password_hash

# app = Flask(__name__)
# app.secret_key = "56f09f32652e417af985d24a9ab63aefb41734cfd5d4e70d"

# app.static_folder = "static"

# # Database connection function
# def get_db_connection():
#     conn = sqlite3.connect("users.db")
#     conn.row_factory = sqlite3.Row
#     return conn

# # Initialize the database
# def init_db():
#     with sqlite3.connect("users.db") as conn:
#         conn.execute('''
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 phone TEXT UNIQUE NOT NULL,
#                 name TEXT NOT NULL,
#                 email TEXT UNIQUE NOT NULL,
#                 age_category TEXT NOT NULL,
#                 gender TEXT NOT NULL,
#                 password TEXT NOT NULL,
#                 user_type TEXT DEFAULT 'patient' NOT NULL
#             )
#         ''')
#     print("Database initialized.")

# # Load the model and preprocessing objects
# def load_model():
#     model = joblib.load("glucose_model.pkl")
#     scaler = joblib.load("glucose_scaler.pkl")
#     label_encoders = joblib.load("glucose_label_encoders.pkl")
#     return model, scaler, label_encoders

# # Load components once during startup
# model, scaler, label_encoders = load_model()

# @app.route("/")
# @app.route("/index")
# def index():
#     """
#     Render the homepage with form inputs for prediction.
#     """
#     genders = label_encoders["Gender"].classes_.tolist()
#     age_groups = label_encoders["Age Group"].classes_.tolist()
#     return render_template("index.html", genders=genders, age_groups=age_groups)

# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         # Extract form data
#         phone = request.form.get("phone")
#         name = request.form.get("name")
#         email = request.form.get("email")
#         age_category = request.form.get("age")  # 'age' to 'age_category'
#         gender = request.form.get("gender")
#         user_type = request.form.get("type")
#         password = request.form.get("password")

#         # Hash password for security
#         hashed_password = generate_password_hash(password)

#         try:
#             # Save data in database
#             with sqlite3.connect("users.db") as conn:
#                 cursor = conn.cursor()
#                 cursor.execute('''
#                     INSERT INTO users (phone, name, email, age_category, gender, password, user_type)
#                     VALUES (?, ?, ?, ?, ?, ?, ?)
#                 ''', (phone, name, email, age_category, gender, hashed_password, user_type))
#                 conn.commit()

#             # Flash success and redirect
#             flash("Signup successful! Please login to continue.", "success")
#             return redirect(url_for("login"))

#         except sqlite3.IntegrityError:
#             # Handle duplicate entries
#             flash("Phone or email already exists. Please use a different one.", "error")
#         except Exception as e:
#             flash(f"An error occurred: {str(e)}", "error")

#     # Render signup form on GET
#     return render_template("signup.html")

# @app.route("/login", methods=["GET", "POST"])
# def login():
#     """
#     Handle user login.
#     """
#     if request.method == "POST":
#         email = request.form.get("email")
#         password = request.form.get("password")

#         conn = get_db_connection()
#         user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
#         conn.close()

#         if user and check_password_hash(user["password"], password):
#             session["user_id"] = user["id"]
#             session["name"] = user["name"]
#             session["user_type"] = user["user_type"]
#             flash("Login successful!", "success")
#             return redirect(url_for("index"))

#         flash("Invalid email or password.", "error")

#     return render_template("login.html")

# @app.route("/logout")
# def logout():
#     """
#     Logout the user and clear session data.
#     """
#     session.clear()
#     flash("You have been logged out.", "info")
#     return redirect(url_for("login"))

# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Handle predictions based on user input.
#     """
#     if "user_id" not in session:
#         return redirect(url_for("login"))

#     try:
#         gender = request.form.get("gender")
#         age_group = request.form.get("age_group")
#         na = float(request.form.get("na"))
#         k = float(request.form.get("k"))
#         cl = float(request.form.get("cl"))

#         input_data = pd.DataFrame({
#             "Gender": [label_encoders["Gender"].transform([gender])[0]],
#             "Age Group": [label_encoders["Age Group"].transform([age_group])[0]],
#             "Na (mmol/L)": [na],
#             "K (mmol/L)": [k],
#             "Cl (mmol/L)": [cl],
#         })

#         input_scaled = scaler.transform(input_data)
#         prediction = model.predict(input_scaled)
#         category = label_encoders["Glucose Category"].inverse_transform(prediction)[0]

#         return render_template(
#             "index.html",
#             genders=label_encoders["Gender"].classes_.tolist(),
#             age_groups=label_encoders["Age Group"].classes_.tolist(),
#             prediction=f"Predicted Glucose Category: {category}"
#         )
#     except Exception as e:
#         return render_template(
#             "index.html",
#             genders=label_encoders["Gender"].classes_.tolist(),
#             age_groups=label_encoders["Age Group"].classes_.tolist(),
#             error=f"An error occurred: {str(e)}"
#         )

# if __name__ == "__main__":
#     init_db()
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import pandas as pd
import joblib
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "56f09f32652e417af985d24a9ab63aefb41734cfd5d4e70d"
app.static_folder = "static"

# Database connection function
def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row  # This allows row access by column name
    return conn

def init_db():
    with sqlite3.connect("users.db") as conn:
        # Create 'users' table with necessary columns if it doesn't exist
        conn.execute(''' 
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                age_category TEXT NOT NULL,
                gender TEXT NOT NULL,
                password TEXT NOT NULL,
                user_type TEXT DEFAULT 'patient' NOT NULL
            )
        ''')

    print("Database initialized or updated.")

# Load the model and preprocessing objects
def load_model():
    model = joblib.load("glucose_model.pkl")
    scaler = joblib.load("glucose_scaler.pkl")
    label_encoders = joblib.load("glucose_label_encoders.pkl")
    return model, scaler, label_encoders

# Load components once during startup
model, scaler, label_encoders = load_model()

@app.route("/")
@app.route("/index")
def index():
    """
    Render the homepage with form inputs for prediction.
    """
    genders = label_encoders["Gender"].classes_.tolist()
    age_groups = label_encoders["Age Group"].classes_.tolist()
    return render_template("index.html", genders=genders, age_groups=age_groups)

@app.route("/signup")
def signup_page():
    """
    Render the signup page.
    """
    return render_template("signup.html")

@app.route("/login")
def login_page():
    """
    Render the login page.
    """
    return render_template("login.html")

@app.route("/login_doctor")
def login_doctor():
    """
    Render the login_doctor page.
    """
    return render_template("login_doctor.html")

@app.route("/about")
def about_page():
    """
    Render the about page.
    """
    return render_template("about.html")

@app.route("/electrolyte")
def electrolyte_page():
    """
    Render the electrolyte page.
    """
    return render_template("electrolyte.html")

@app.route("/signup", methods=["POST"])
def signup():
    if request.method == "POST":
        # Extract JSON data
        data = request.get_json()
        print(data)

        # Get values from JSON data with proper handling if keys do not exist
        phone = data.get("phone")
        name = data.get("name")
        email = data.get("email")
        age_category = data.get("age")
        gender = data.get("gender")
        user_type = data.get("user_type")
        password = data.get("password")

        # Ensure that essential data exists in the request
        if not all([phone, name, email, age_category, gender, password, user_type]):
            return jsonify({"message": "Missing required fields. Please check your input."}), 400

        # Hash password for security
        hashed_password = generate_password_hash(password)

        try:
            # Save data into the database
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute(''' 
                    INSERT INTO users (phone, name, email, age_category, gender, password, user_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (phone, name, email, age_category, gender, hashed_password, user_type))
                conn.commit()

            return jsonify({"message": "Signup successful! Please login to continue."}), 201

        except sqlite3.IntegrityError:
            return jsonify({"message": "Phone or email already exists. Please use a different one."}), 400
        except Exception as e:
            return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route("/login", methods=["POST"])
def login():
    """
    Handle user login.
    """
    if request.method == "POST":
        # Extract JSON data
        data = request.get_json()

        email = data.get("email")
        password = data.get("password")

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user["password"], password):
            # User details stored in session upon successful login
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["email"] = user["email"]
            session["user_type"] = user["user_type"]

            return jsonify({"message": "Login successful!"}), 200

        conn.close()
        return jsonify({"message": "Invalid email or password."}), 400


@app.route("/logout", methods=["POST"])
def logout():
    """
    Handle user logout.
    """
    session.pop("user_id", None)  # Remove user session
    return jsonify({"message": "Logged out successfully"}), 200

@app.route("/predict", methods=["GET", "POST"])
def electrolyte():
    """
    Render the electrolyte page and handle diabetes type predictions based on input.
    """
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            # Extract JSON data from the POST request
            data = request.get_json()

            if not data:
                return jsonify({"error": "No input data provided"}), 400

            gender = data.get("gender")
            age_group = data.get("age_group")
            na = data.get("na")
            k = data.get("k")
            cl = data.get("cl")

            if not all([gender, age_group, na, k, cl]):
                return jsonify({"error": "Missing one or more input fields"}), 400

            # Prepare input data for prediction (convert categorical inputs using label encoding)
            input_data = pd.DataFrame({
                "Gender": [label_encoders["Gender"].transform([gender])[0]],
                "Age Group": [label_encoders["Age Group"].transform([age_group])[0]],
                "Na (mmol/L)": [na],
                "K (mmol/L)": [k],
                "Cl (mmol/L)": [cl],
            })

            # Scale the input data
            input_scaled = scaler.transform(input_data)

            # Make the prediction
            prediction = model.predict(input_scaled)

            # Decode the prediction category (diabetes prediction)
            category = label_encoders["Glucose Category"].inverse_transform(prediction)[0]

            # Return the prediction result as JSON
            return jsonify({"prediction": category}), 200

        except Exception as e:
            # Handle errors and return the error message as JSON
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    # Default behavior for GET requests: render the page (form)
    return render_template("electrolyte.html")


# @app.route("/profile", methods=["GET"])
# def profile():
#     """
#     Render the profile page for the logged-in user.
#     """
#     if "user_id" not in session:
#         return redirect(url_for("login"))

#     try:
#         # Fetch user details from database (assuming a 'users' table exists)
#         user_id = session["user_id"]
#         user = db.execute("SELECT name, email, phone, gender FROM users WHERE id = ?", (user_id,)).fetchone()

#         if user:
#             diabetes_type = session.get("diabetes_type", "Not Predicted Yet")
#             return render_template(
#                 "profile.html",
#                 user_name=user["name"],
#                 email=user["email"],
#                 phone=user["phone"],
#                 gender=user["gender"],
#                 diabetes_type=diabetes_type
#             )
#         else:
#             return redirect(url_for("login"))

#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return redirect(url_for("login"))
@app.route("/profile", methods=["GET"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute("SELECT name, email, phone, gender FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()

    if user:
        # Convert sqlite3.Row to dictionary for easier debugging and template usage
        user_dict = dict(user)

        # Check if the data is fetched correctly
        print(user_dict)

        # Pass to the template
        diabetes_type = session.get("diabetes_type", "Not Predicted Yet")
        return render_template(
            "profile.html",
            **user_dict,  # Unpack user dictionary into individual template variables
            diabetes_type=diabetes_type,
        )

    return redirect(url_for("login"))

@app.route("/doctors")
def doctors():
    return render_template("doctor.html")



if __name__ == "__main__":
    init_db()
    app.run(debug=True)




