import 'congestion.dart';

class TransitResponse {
  final int totalMinutes;
  final int stops;
  final String summary;
  final List<NextTrain> nextTrains;
  final List<double> todaysOccupancy;
  final String recommendation;

  const TransitResponse({
    required this.totalMinutes,
    required this.stops,
    required this.summary,
    required this.nextTrains,
    required this.todaysOccupancy,
    required this.recommendation,
  });

  factory TransitResponse.fromJson(Map<String, dynamic> json) {
    return TransitResponse(
      totalMinutes: json['total_minutes'] as int,
      stops: json['stops'] as int,
      summary: json['summary'] as String,
      nextTrains: (json['next_trains'] as List)
          .map((e) => NextTrain.fromJson(e as Map<String, dynamic>))
          .toList(),
      todaysOccupancy: (json['todays_occupancy'] as List)
          .map((e) => (e as num).toDouble())
          .toList(),
      recommendation: json['recommendation'] as String,
    );
  }
}

class NextTrain {
  final String trainLine;
  final int arrivalMinutes;
  final int platform;
  final double congestionValue;
  final int congestionCode;
  final String congestionLevel;

  const NextTrain({
    required this.trainLine,
    required this.arrivalMinutes,
    required this.platform,
    required this.congestionValue,
    required this.congestionCode,
    required this.congestionLevel,
  });

  factory NextTrain.fromJson(Map<String, dynamic> json) {
    return NextTrain(
      trainLine: json['train_line'] as String,
      arrivalMinutes: json['arrival_minutes'] as int,
      platform: json['platform'] as int,
      congestionValue: (json['congestion_value'] as num).toDouble(),
      congestionCode: json['congestion_code'] as int,
      congestionLevel: json['congestion_level'] as String,
    );
  }

  CongestionLevel get congestion {
    final percentage = (congestionValue * 100).toInt();
    return CongestionLevelExtension.fromPercentage(percentage);
  }

  String get arrivalText {
    final hours = arrivalMinutes ~/ 60;
    final mins = arrivalMinutes % 60;
    if (hours > 0) {
      return '${hours}h ${mins}m';
    }
    return '$mins min';
  }

  String get arrivalTimeText {
    final now = DateTime.now();
    final arrival = now.add(Duration(minutes: arrivalMinutes));
    return '${arrival.hour.toString().padLeft(2, '0')}:${arrival.minute.toString().padLeft(2, '0')}';
  }
}
