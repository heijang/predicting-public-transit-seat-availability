import 'congestion.dart';

class TrainArrival {
  final int arrivalMinutes;
  final String destination;
  final CongestionLevel congestion;
  final String trainNumber;

  const TrainArrival({
    required this.arrivalMinutes,
    required this.destination,
    required this.congestion,
    required this.trainNumber,
  });

  String get arrivalText {
    if (arrivalMinutes == 0) {
      return 'Now';
    } else if (arrivalMinutes == 1) {
      return '1 min';
    } else {
      return '$arrivalMinutes min';
    }
  }
}
