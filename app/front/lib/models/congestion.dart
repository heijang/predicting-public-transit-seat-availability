import 'package:flutter/material.dart';

enum CongestionLevel {
  comfortable, // 0-50%
  normal, // 51-80%
  crowded, // 81-100%
}

extension CongestionLevelExtension on CongestionLevel {
  String get label {
    switch (this) {
      case CongestionLevel.comfortable:
        return 'Low';
      case CongestionLevel.normal:
        return 'Medium';
      case CongestionLevel.crowded:
        return 'High';
    }
  }

  Color get color {
    switch (this) {
      case CongestionLevel.comfortable:
        return Colors.green;
      case CongestionLevel.normal:
        return Colors.orange;
      case CongestionLevel.crowded:
        return Colors.red;
    }
  }

  String get percentageRange {
    switch (this) {
      case CongestionLevel.comfortable:
        return '0-50%';
      case CongestionLevel.normal:
        return '51-80%';
      case CongestionLevel.crowded:
        return '81-100%';
    }
  }

  static CongestionLevel fromPercentage(int percentage) {
    if (percentage <= 50) {
      return CongestionLevel.comfortable;
    } else if (percentage <= 80) {
      return CongestionLevel.normal;
    } else {
      return CongestionLevel.crowded;
    }
  }
}
