import 'package:flutter/material.dart';
import '../models/congestion.dart';

class CongestionBadge extends StatelessWidget {
  final CongestionLevel congestion;
  final bool showPercentage;

  const CongestionBadge({
    super.key,
    required this.congestion,
    this.showPercentage = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: congestion.color,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        showPercentage
            ? '${congestion.label} (${congestion.percentageRange})'
            : congestion.label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 9,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}
