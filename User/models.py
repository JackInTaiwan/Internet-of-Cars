from django.db import models



class User(models.Model) :
    user_id = models.IntegerField(max_length=50)
    pos_x = models.FloatField(max_length=15, default=0.0)
    pos_y = models.FloatField(max_length=15, default=0.0)
    create_time = models.DateTimeField(auto_now_add=True)
    v = models.FloatField(max_length=10, default=0.0)

    class Meta :
        ordering = ["create_time"]
        unique_together = ["user_id",]