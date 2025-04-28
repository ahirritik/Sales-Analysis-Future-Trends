from django import template

register = template.Library()

@register.filter
def index(sequence, position):
    try:
        return sequence[position]
    except (IndexError, TypeError):
        return None

@register.filter
def zip(list1, list2):
    return zip(list1, list2) 