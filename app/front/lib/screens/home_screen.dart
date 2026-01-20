import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/station.dart';
import '../models/train_arrival.dart';
import '../data/station_data.dart';
import '../services/location_service.dart';
import '../widgets/location_display.dart';
import '../widgets/station_selector.dart';
import '../widgets/line_map_view.dart';
import '../widgets/train_arrival_card.dart';
import '../widgets/congestion_graph.dart';

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
  List<TrainArrival> _arrivals = [];

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
        _arrivals = [];
      }
    });
  }

  // Ensure stations are in forward direction (lower index -> higher index)
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
      _arrivals = StationData.getArrivalsForStation(_departureStation!.stopId);
    }
  }

  void _searchRoute() {
    if (_departureStation != null && _arrivalStation != null) {
      setState(() {
        _updateRoute();
      });
    }
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
                  if (_route.isNotEmpty) ...[
                    SizedBox(height: isMobile ? 6 : 12),
                    _buildRouteInfo(isMobile),
                    SizedBox(height: isMobile ? 6 : 12),
                    _buildArrivalsSection(isMobile),
                    SizedBox(height: isMobile ? 6 : 12),
                    const CongestionGraph(),
                  ],
                ],
              ),
            ),
          ),
        ),
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
                            _arrivalStation != null
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
                    child: const Text(
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
                        if (_departureStation != null) _updateRoute();
                      });
                    },
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed:
                      _departureStation != null && _arrivalStation != null
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
                  child: const Text(
                    'Search',
                    style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
    );
  }

  Widget _buildRouteInfo(bool isMobile) {
    final travelTime = StationData.getEstimatedTravelTime(
      _departureStation!.stopId,
      _arrivalStation!.stopId,
    );

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
            '~$travelTime min (${_route.length} stops)',
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
        ..._arrivals.asMap().entries.map((entry) {
          return TrainArrivalCard(
            arrival: entry.value,
            index: entry.key,
          );
        }),
      ],
    );
  }
}
