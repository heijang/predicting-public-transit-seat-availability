import 'dart:math';
import 'package:flutter/material.dart';
import '../models/station.dart';
import '../models/train_arrival.dart';
import '../models/congestion.dart';

class StationData {
  static const String lineName = 'RE5';
  static const Color lineColor = Color(0xFFE30613); // DB Red

  // Berlin RE5 major stations
  static const List<Map<String, dynamic>> _stationList = [
    {'stopId': 'RE5-01', 'name': 'Stralsund'},
    {'stopId': 'RE5-02', 'name': 'Neubrandenburg'},
    {'stopId': 'RE5-03', 'name': 'Neustrelitz'},
    {'stopId': 'RE5-04', 'name': 'Oranienburg'},
    {'stopId': 'RE5-05', 'name': 'Berlin Hbf'},
    {'stopId': 'RE5-06', 'name': 'Berlin Süd'},
    {'stopId': 'RE5-07', 'name': 'Wünsdorf'},
    {'stopId': 'RE5-08', 'name': 'Elsterwerda'},
  ];

  static List<Station> getStations() {
    return _stationList.asMap().entries.map((entry) {
      return Station(
        stopId: entry.value['stopId'] as String,
        name: entry.value['name'] as String,
        orderIndex: entry.key,
        arrivals: _generateMockArrivals(),
      );
    }).toList();
  }

  static Station? getStationById(String stopId) {
    final stations = getStations();
    try {
      return stations.firstWhere((s) => s.stopId == stopId);
    } catch (_) {
      return null;
    }
  }

  static List<Station> getRouteBetween(String fromStopId, String toStopId) {
    final stations = getStations();
    final fromIndex = stations.indexWhere((s) => s.stopId == fromStopId);
    final toIndex = stations.indexWhere((s) => s.stopId == toStopId);

    if (fromIndex == -1 || toIndex == -1) return [];

    if (fromIndex <= toIndex) {
      return stations.sublist(fromIndex, toIndex + 1);
    } else {
      return stations.sublist(toIndex, fromIndex + 1).reversed.toList();
    }
  }

  static int getEstimatedTravelTime(String fromStopId, String toStopId) {
    final route = getRouteBetween(fromStopId, toStopId);
    return (route.length - 1) * 15; // ~15 min between major stations
  }

  static List<TrainArrival> _generateMockArrivals() {
    final random = Random();
    final destinations = ['Stralsund', 'Wünsdorf', 'Elsterwerda', 'Rostock'];
    final destination = destinations[random.nextInt(destinations.length)];

    return [
      TrainArrival(
        arrivalMinutes: random.nextInt(5) + 1,
        destination: destination,
        congestion: CongestionLevelExtension.fromPercentage(
          random.nextInt(50),
        ),
        trainNumber: 'RE${random.nextInt(90) + 10}',
      ),
      TrainArrival(
        arrivalMinutes: random.nextInt(10) + 8,
        destination: destination,
        congestion: CongestionLevelExtension.fromPercentage(
          random.nextInt(40) + 40,
        ),
        trainNumber: 'RE${random.nextInt(90) + 10}',
      ),
      TrainArrival(
        arrivalMinutes: random.nextInt(10) + 20,
        destination: destination,
        congestion: CongestionLevelExtension.fromPercentage(
          random.nextInt(30) + 70,
        ),
        trainNumber: 'RE${random.nextInt(90) + 10}',
      ),
    ];
  }

  static List<TrainArrival> getArrivalsForStation(String stopId) {
    return _generateMockArrivals();
  }

  // Daily congestion data (hourly from 5:00 to 23:00)
  static List<int> getDailyCongestionData() {
    return [
      25, 35, 55, 85, 90, 75, 50, 45, 40, 45,
      55, 70, 65, 55, 60, 75, 85, 70, 45
    ];
  }

  static List<String> getHourLabels() {
    return ['5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
            '15', '16', '17', '18', '19', '20', '21', '22', '23'];
  }
}
