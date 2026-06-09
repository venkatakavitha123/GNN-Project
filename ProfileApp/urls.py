from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
		     path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
		     path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
	             path("UserLogin.html", views.UserLogin, name="UserLogin"),
		     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		     path("Register.html", views.Register, name="Register"),
		     path("RegisterAction", views.RegisterAction, name="RegisterAction"),
		     path("LoadDataset.html", views.LoadDataset, name="LoadDataset"),
		     path("LoadDatasetAction", views.LoadDatasetAction, name="LoadDatasetAction"),
		     path("RunGNN", views.RunGNN, name="RunGNN"),
		     path("ViewProfile", views.ViewProfile, name="ViewProfile"),
		     path("Predict", views.Predict, name="Predict"),
		     path("PredictAction", views.PredictAction, name="PredictAction"),		     
		    ]