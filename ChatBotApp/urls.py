from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
                        path("Upload.html", views.Upload, name="Upload"),
	                path("UploadAction", views.UploadAction, name="UploadAction"),	
			path("Record", views.Record, name="Record"),	
			path("ChatData", views.ChatData, name="ChatData"),
			path("record", views.record, name="record"),
]