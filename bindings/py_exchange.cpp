// bindings/py_exchange.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "matching_engine.hpp"
#include "multi_asset_engine.hpp"

#include <string>

namespace py = pybind11;
using namespace exsim;

PYBIND11_MODULE(exchange_simulator, m) {
    m.doc() = "Python bindings for the C++ exchange simulator";

    // Enums
    py::enum_<Side>(m, "Side")
        .value("Buy", Side::Buy)
        .value("Sell", Side::Sell);

    py::enum_<OrderType>(m, "OrderType")
        .value("Limit", OrderType::Limit)
        .value("Market", OrderType::Market)
        .value("Stop", OrderType::Stop)
        .value("StopLimit", OrderType::StopLimit)
        .value("Pegged", OrderType::Pegged);

    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("GTC", TimeInForce::GTC)
        .value("IOC", TimeInForce::IOC)
        .value("FOK", TimeInForce::FOK);

    py::enum_<Phase>(m, "Phase")
        .value("PreOpen", Phase::PreOpen)
        .value("Continuous", Phase::Continuous);

    py::enum_<CancelResult>(m, "CancelResult")
        .value("Success", CancelResult::Success)
        .value("OrderNotFound", CancelResult::OrderNotFound)
        .value("AlreadyFilled", CancelResult::AlreadyFilled);

    // Order struct
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("id", &Order::id)
        .def_readwrite("price", &Order::price)
        .def_readwrite("stop_price", &Order::stop_price)
        .def_readwrite("peg_offset", &Order::peg_offset)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("filled_quantity", &Order::filled_quantity)
        .def_readwrite("visible_quantity", &Order::visible_quantity)
        .def_readwrite("hidden_quantity", &Order::hidden_quantity)
        .def_readwrite("side", &Order::side)
        .def_readwrite("type", &Order::type)
        .def_readwrite("tif", &Order::tif)
        .def_readwrite("timestamp", &Order::timestamp)
        .def("remaining", &Order::remaining)
        .def("is_filled", &Order::is_filled)
        .def("is_iceberg", &Order::is_iceberg);

    // Fill struct
    py::class_<Fill>(m, "Fill")
        .def_readonly("maker_order_id", &Fill::maker_order_id)
        .def_readonly("taker_order_id", &Fill::taker_order_id)
        .def_readonly("price", &Fill::price)
        .def_readonly("quantity", &Fill::quantity)
        .def_readonly("aggressor_side", &Fill::aggressor_side)
        .def_readonly("timestamp", &Fill::timestamp);

    // One price level of an L2 snapshot
    py::class_<LevelView>(m, "LevelView")
        .def_readonly("price", &LevelView::price)
        .def_readonly("total_quantity", &LevelView::total_quantity)
        .def_readonly("order_count", &LevelView::order_count)
        .def("__repr__", [](const LevelView& l) {
            return "<LevelView price=" + std::to_string(l.price) +
                   " qty=" + std::to_string(l.total_quantity) +
                   " orders=" + std::to_string(l.order_count) + ">";
        });

    // OrderBook (read-only access via MatchingEngine::book())
    py::class_<OrderBook>(m, "OrderBook")
        .def("bid_depth", &OrderBook::bid_depth,
             "Number of distinct bid price levels. Not a quantity.")
        .def("ask_depth", &OrderBook::ask_depth,
             "Number of distinct ask price levels. Not a quantity.")
        .def("l2_bids", &OrderBook::l2_bids, py::arg("max_levels") = 5,
             "Bid levels from the touch outwards, best first.")
        .def("l2_asks", &OrderBook::l2_asks, py::arg("max_levels") = 5,
             "Ask levels from the touch outwards, best first.")
        .def("spread", &OrderBook::spread)
        .def("best_bid_price", [](const OrderBook& book) -> py::object {
            auto* level = book.best_bid();
            if (!level) return py::none();
            return py::cast(level->price);
        })
        .def("best_ask_price", [](const OrderBook& book) -> py::object {
            auto* level = book.best_ask();
            if (!level) return py::none();
            return py::cast(level->price);
        });

    // MatchingEngine
    py::class_<MatchingEngine>(m, "MatchingEngine")
        .def(py::init<>())
        .def("submit", &MatchingEngine::submit)
        .def("cancel", &MatchingEngine::cancel)
        .def("uncross", &MatchingEngine::uncross)
        .def("set_phase", &MatchingEngine::set_phase)
        .def("phase", &MatchingEngine::phase)
        .def("book", &MatchingEngine::book, py::return_value_policy::reference_internal)
        .def("stop_order_count", &MatchingEngine::stop_order_count)
        .def("pegged_order_count", &MatchingEngine::pegged_order_count);

    // MultiAssetEngine
    py::class_<MultiAssetEngine>(m, "MultiAssetEngine")
        .def(py::init<>())
        .def("get_book", &MultiAssetEngine::get_book, py::return_value_policy::reference_internal)
        .def("submit", &MultiAssetEngine::submit, py::arg("symbol_id"), py::arg("order"))
        .def("cancel", &MultiAssetEngine::cancel, py::arg("order_id"))
        .def("symbols", &MultiAssetEngine::symbols);
}
