import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/station.dart';
import '../models/transit_response.dart';
import '../data/station_data.dart';
import '../services/location_service.dart';
import '../services/transit_api_service.dart';
import '../widgets/location_display.dart';
import '../widgets/station_selector.dart';
import '../widgets/line_map_view.dart';
import '../widgets/congestion_graph.dart';
import '../widgets/congestion_badge.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late List<Station> _stations;
  Station? _departureStation;
  Station? _arrivalStation;
  List<Station> _route = [];
  TransitResponse? _transitResponse;
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _stations = StationData.getStations();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<LocationService>().getCurrentLocation();
    });
  }

  void _onStationTap(Station station) {
    setState(() {
      if (_departureStation == null) {
        _departureStation = station;
      } else if (_arrivalStation == null) {
        if (station.stopId != _departureStation!.stopId) {
          _arrivalStation = station;
          _ensureForwardDirection();
          _updateRoute();
        }
      } else {
        _departureStation = station;
        _arrivalStation = null;
        _route = [];
        _transitResponse = null;
        _errorMessage = null;
      }
    });
  }

  void _ensureForwardDirection() {
    if (_departureStation != null && _arrivalStation != null) {
      if (_departureStation!.orderIndex > _arrivalStation!.orderIndex) {
        final temp = _departureStation;
        _departureStation = _arrivalStation;
        _arrivalStation = temp;
      }
    }
  }

  void _updateRoute() {
    if (_departureStation != null && _arrivalStation != null) {
      _ensureForwardDirection();
      _route = StationData.getRouteBetween(
        _departureStation!.stopId,
        _arrivalStation!.stopId,
      );
    }
  }

  Future<void> _searchRoute() async {
    if (_departureStation == null || _arrivalStation == null) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _updateRoute();
    });

    final response = await TransitApiService.getTransitInfo(
      departureStation: _departureStation!.stopId,
      arrivalStation: _arrivalStation!.stopId,
      departureTime: TransitApiService.getCurrentTime(),
    );

    setState(() {
      _isLoading = false;
      if (response != null) {
        _transitResponse = response;
      } else {
        _errorMessage = 'Failed to fetch transit info. Please try again.';
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final isMobile = screenWidth < 600;

    return Scaffold(
      backgroundColor: const Color(0xFFF5F6FA),
      appBar: AppBar(
        backgroundColor: StationData.lineColor,
        foregroundColor: Colors.white,
        toolbarHeight: isMobile ? 44 : 56,
        title: Text(
          'Train Navigator',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: isMobile ? 14 : 18,
          ),
        ),
        actions: const [
          Padding(
            padding: EdgeInsets.only(right: 8),
            child: LocationDisplay(),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(isMobile ? 8 : 16),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 800),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildStationSelectors(isMobile),
                  SizedBox(height: isMobile ? 6 : 12),
                  LineMapView(
                    stations: _stations,
                    departureStation: _departureStation,
                    arrivalStation: _arrivalStation,
                    onStationTap: _onStationTap,
                  ),
                  if (_isLoading) ...[
                    SizedBox(height: isMobile ? 6 : 12),
                    const Center(
                      child: CircularProgressIndicator(),
                    ),
                  ] else if (_errorMessage != null) ...[
                    SizedBox(height: isMobile ? 6 : 12),
                    _buildErrorMessage(isMobile),
                  ] else if (_route.isNotEmpty && _transitResponse != null) ...[
                    SizedBox(height: isMobile ? 6 : 12),
                    _buildRouteInfo(isMobile),
                    SizedBox(height: isMobile ? 6 : 12),
                    _buildArrivalsSection(isMobile),
                    SizedBox(height: isMobile ? 6 : 12),
                    CongestionGraph(
                      occupancyData: _transitResponse!.todaysOccupancy,
                      recommendation: _transitResponse!.recommendation,
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildErrorMessage(bool isMobile) {
    return Container(
      padding: EdgeInsets.all(isMobile ? 8 : 12),
      decoration: BoxDecoration(
        color: Colors.red[50],
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.red[200]!),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: Colors.red[700], size: isMobile ? 16 : 20),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _errorMessage!,
              style: TextStyle(fontSize: isMobile ? 11 : 13, color: Colors.red[700]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStationSelectors(bool isMobile) {
    return Container(
      padding: EdgeInsets.all(isMobile ? 8 : 12),
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
      child: isMobile
          ? Column(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: StationSelector(
                        label: 'From',
                        selectedStation: _departureStation,
                        stations: _stations,
                        onChanged: (station) {
                          setState(() {
                            _departureStation = station;
                            _transitResponse = null;
                            _errorMessage = null;
                            if (_arrivalStation != null) _updateRoute();
                          });
                        },
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4),
                      child: Icon(
                        Icons.arrow_forward,
                        size: 18,
                        color: StationData.lineColor,
                      ),
                    ),
                    Expanded(
                      child: StationSelector(
                        label: 'To',
                        selectedStation: _arrivalStation,
                        stations: _stations,
                        onChanged: (station) {
                          setState(() {
                            _arrivalStation = station;
                            _transitResponse = null;
                            _errorMessage = null;
                            if (_departureStation != null) _updateRoute();
                          });
                        },
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  height: 32,
                  child: ElevatedButton(
                    onPressed: _departureStation != null &&
                            _arrivalStation != null &&
                            !_isLoading
                        ? _searchRoute
                        : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: StationData.lineColor,
                      foregroundColor: Colors.white,
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(6),
                      ),
                    ),
                    child: _isLoading
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Text(
                            'Search Route',
                            style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                          ),
                  ),
                ),
              ],
            )
          : Row(
              children: [
                Expanded(
                  child: StationSelector(
                    label: 'From',
                    selectedStation: _departureStation,
                    stations: _stations,
                    onChanged: (station) {
                      setState(() {
                        _departureStation = station;
                        _transitResponse = null;
                        _errorMessage = null;
                        if (_arrivalStation != null) _updateRoute();
                      });
                    },
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: Icon(
                    Icons.arrow_forward,
                    color: StationData.lineColor,
                  ),
                ),
                Expanded(
                  child: StationSelector(
                    label: 'To',
                    selectedStation: _arrivalStation,
                    stations: _stations,
                    onChanged: (station) {
                      setState(() {
                        _arrivalStation = station;
                        _transitResponse = null;
                        _errorMessage = null;
                        if (_departureStation != null) _updateRoute();
                      });
                    },
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed:
                      _departureStation != null && _arrivalStation != null && !_isLoading
                          ? _searchRoute
                          : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: StationData.lineColor,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 12,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                    ),
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Text(
                          'Search',
                          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                        ),
                ),
              ],
            ),
    );
  }

  Widget _buildRouteInfo(bool isMobile) {
    return Container(
      padding: EdgeInsets.all(isMobile ? 8 : 12),
      decoration: BoxDecoration(
        color: StationData.lineColor.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: StationData.lineColor.withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        children: [
          Icon(Icons.route, color: StationData.lineColor, size: isMobile ? 16 : 20),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              '${_departureStation!.name} → ${_arrivalStation!.name}',
              style: TextStyle(
                fontSize: isMobile ? 12 : 14,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          Text(
            _transitResponse!.summary,
            style: TextStyle(
              fontSize: isMobile ? 10 : 12,
              color: Colors.grey[700],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildArrivalsSection(bool isMobile) {
    final nextTrains = _transitResponse!.nextTrains;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.train, color: StationData.lineColor, size: isMobile ? 14 : 18),
            const SizedBox(width: 6),
            Text(
              'Next Trains',
              style: TextStyle(
                fontSize: isMobile ? 12 : 16,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        SizedBox(height: isMobile ? 6 : 10),
        ...nextTrains.asMap().entries.map((entry) {
          return _buildTrainCard(entry.value, entry.key, isMobile);
        }),
      ],
    );
  }

  Widget _buildTrainCard(NextTrain train, int index, bool isMobile) {
    return Container(
      margin: const EdgeInsets.only(bottom: 4),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 3,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(
              color: _getIndexColor(index),
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                '${index + 1}',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 11,
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      train.arrivalTimeText,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      '(${train.arrivalText})',
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.grey[600],
                      ),
                    ),
                    const SizedBox(width: 6),
                    CongestionBadge(congestion: train.congestion),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  '${train.trainLine} · Platform ${train.platform}',
                  style: TextStyle(fontSize: 10, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getIndexColor(int index) {
    switch (index) {
      case 0:
        return Colors.blue;
      case 1:
        return Colors.indigo;
      case 2:
        return Colors.deepPurple;
      default:
        return Colors.grey;
    }
  }
}
