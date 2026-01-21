import 'package:flutter/material.dart';
import '../data/station_data.dart';
import '../models/congestion.dart';

class CongestionGraph extends StatelessWidget {
  final List<double> occupancyData;
  final String? recommendation;

  const CongestionGraph({
    super.key,
    required this.occupancyData,
    this.recommendation,
  });

  @override
  Widget build(BuildContext context) {
    // Convert 0.0-1.0 to 0-100 percentage
    final data = occupancyData.map((e) => (e * 100).toInt()).toList();
    final labels = _generateHourLabels();
    final currentHour = DateTime.now().hour;
    final currentIndex = currentHour < data.length ? currentHour : -1;

    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 4,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.show_chart, size: 14, color: StationData.lineColor),
              const SizedBox(width: 4),
              const Text(
                'Today\'s Congestion',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              _buildLegend(),
            ],
          ),
          const SizedBox(height: 6),
          SizedBox(
            height: 60,
            child: CustomPaint(
              size: const Size(double.infinity, 60),
              painter: _GraphPainter(
                data: data,
                currentHourIndex: currentIndex,
              ),
            ),
          ),
          const SizedBox(height: 2),
          _buildHourLabels(labels),
          const SizedBox(height: 8),
          _buildRecommendationBox(),
        ],
      ),
    );
  }

  List<String> _generateHourLabels() {
    return List.generate(24, (i) => i.toString());
  }

  Widget _buildLegend() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _legendItem(Colors.green, 'Low'),
        const SizedBox(width: 6),
        _legendItem(Colors.orange, 'Med'),
        const SizedBox(width: 6),
        _legendItem(Colors.red, 'High'),
      ],
    );
  }

  Widget _legendItem(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 6,
          height: 6,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 2),
        Text(label, style: const TextStyle(fontSize: 8, color: Colors.grey)),
      ],
    );
  }

  Widget _buildHourLabels(List<String> labels) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        const Text('0', style: TextStyle(fontSize: 8, color: Colors.grey)),
        const Text('12', style: TextStyle(fontSize: 8, color: Colors.grey)),
        const Text('23', style: TextStyle(fontSize: 8, color: Colors.grey)),
      ],
    );
  }

  Widget _buildRecommendationBox() {
    final displayText = recommendation ?? 'Select a route to see recommendations.';

    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.grey[200]!),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.lightbulb_outline, size: 14, color: Colors.amber[700]),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              displayText,
              style: TextStyle(fontSize: 11, color: Colors.grey[800]),
            ),
          ),
        ],
      ),
    );
  }
}

class _GraphPainter extends CustomPainter {
  final List<int> data;
  final int currentHourIndex;

  _GraphPainter({required this.data, required this.currentHourIndex});

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final paint = Paint()
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final fillPaint = Paint()..style = PaintingStyle.fill;

    final stepX = size.width / (data.length - 1);
    final maxY = size.height - 4;

    // Draw gradient fill
    final path = Path();
    path.moveTo(0, maxY);

    for (var i = 0; i < data.length; i++) {
      final x = i * stepX;
      final y = maxY - (data[i] / 100 * (maxY - 4));
      if (i == 0) {
        path.lineTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    path.lineTo(size.width, maxY);
    path.close();

    fillPaint.shader = LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [
        StationData.lineColor.withValues(alpha: 0.3),
        StationData.lineColor.withValues(alpha: 0.05),
      ],
    ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    canvas.drawPath(path, fillPaint);

    // Draw line segments with color based on congestion
    for (var i = 0; i < data.length - 1; i++) {
      final x1 = i * stepX;
      final y1 = maxY - (data[i] / 100 * (maxY - 4));
      final x2 = (i + 1) * stepX;
      final y2 = maxY - (data[i + 1] / 100 * (maxY - 4));

      final avgValue = (data[i] + data[i + 1]) / 2;
      paint.color = CongestionLevelExtension.fromPercentage(avgValue.toInt()).color;

      canvas.drawLine(Offset(x1, y1), Offset(x2, y2), paint);
    }

    // Draw current hour indicator
    if (currentHourIndex >= 0 && currentHourIndex < data.length) {
      final x = currentHourIndex * stepX;
      final y = maxY - (data[currentHourIndex] / 100 * (maxY - 4));

      final dotPaint = Paint()
        ..color = StationData.lineColor
        ..style = PaintingStyle.fill;

      canvas.drawCircle(Offset(x, y), 4, dotPaint);

      final borderPaint = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5;

      canvas.drawCircle(Offset(x, y), 4, borderPaint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
