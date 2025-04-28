from django.db import models

class SalesData(models.Model):
    id = models.AutoField(primary_key=True)
    day = models.DateField()
    sales = models.IntegerField()
    product = models.CharField(max_length=50)
    price = models.FloatField()
    
    class Meta:
        db_table = 'sales_data'
        indexes = [
            models.Index(fields=['day']),
            models.Index(fields=['product']),
        ]
        
    def __str__(self):
        return f"{self.day} - {self.product} - {self.sales}" 