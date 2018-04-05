from django.db import models



class Event(models.Model) :
    user_id = models.IntegerField(max_length=50)
    pos_x = models.FloatField(max_length=15, default=0.0)
    pos_y = models.FloatField(max_length=15, default=0.0)
    title = models.CharField(max_length=100, default="")
    time = models.DateTimeField(auto_now_add=True)

    class Meta :
        ordering = ["time"]

    def toDict(self) :
        output_dict = dict()
        output_dict["user_id"] = self.user_id
        output_dict["pos_x"] = self.pos_x
        output_dict["pos_y"] = self.pos_y
        output_dict["title"] = self.title
        output_dict["time"] = self.time.strftime("%Y-%m-%d %H:%M:%S")

        return output_dict
