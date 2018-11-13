from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import collections
import json
import logging
import os

def serve_recommendations(request):