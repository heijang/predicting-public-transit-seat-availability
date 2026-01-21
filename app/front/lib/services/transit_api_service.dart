import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/transit_response.dart';

class TransitApiService {
  static const String _baseUrl = 'http://localhost:8000';

  static Future<TransitResponse?> getTransitInfo({
    required String departureStation,
    required String arrivalStation,
    required String departureTime,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/transit'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'departure_station': departureStation,
          'arrival_station': arrivalStation,
          'departure_time': departureTime,
        }),
      );

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body) as Map<String, dynamic>;
        return TransitResponse.fromJson(json);
      } else {
        debugPrint('API Error: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      debugPrint('API Exception: $e');
      return null;
    }
  }

  static String getCurrentTime() {
    final now = DateTime.now();
    return '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}';
  }
}
