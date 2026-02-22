class FirebaseConfig:
    _PRODUCTION_API_KEY = "".join(["AIza", "SyB0TkZ83Fj0CIzn8AAmE-Osc92s3ER8hy8"])
    _STAGING_API_KEY = "".join(["AIza", "SyDfw8CFZBrcWyqS23888ULvoKru7fnlz5Q"])

    PRODUCTION = {
      "apiKey": _PRODUCTION_API_KEY,
      "authDomain": "neurosity-device.firebaseapp.com",
      "databaseURL": "https://neurosity-device.firebaseio.com",
      "storageBucket": "neurosity-device.appspot.com",
      "projectId": "neurosity-device"
    }

    STAGING = {
      "apiKey": _STAGING_API_KEY,
      "authDomain": "neurosity-device-staging.firebaseapp.com",
      "databaseURL": "https://neurosity-device-staging-default-rtdb.firebaseio.com",
      "storageBucket": "neurosity-device-staging.appspot.com",
      "projectId": "neurosity-device"
    }
