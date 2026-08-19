// engine/tests/test_order_book.cpp
#include <gtest/gtest.h>
#include "order_book.hpp"

using namespace exsim;

class OrderBookTest : public ::testing::Test {
protected:
    OrderBook book;
    uint64_t next_id = 1;

    Order make_limit(Side side, Price price, Quantity qty) {
        Order o{};
        o.id = next_id++;
        o.side = side;
        o.price = price;
        o.quantity = qty;
        o.filled_quantity = 0;
        o.type = OrderType::Limit;
        o.tif = TimeInForce::GTC;
        o.timestamp = next_id; // increasing
        return o;
    }
};

TEST_F(OrderBookTest, EmptyBook) {
    EXPECT_EQ(book.best_bid(), nullptr);
    EXPECT_EQ(book.best_ask(), nullptr);
    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_EQ(book.ask_depth(), 0);
}

TEST_F(OrderBookTest, AddBidOrder) {
    auto order = make_limit(Side::Buy, 10000, 50);
    EXPECT_TRUE(book.add(order));
    ASSERT_NE(book.best_bid(), nullptr);
    EXPECT_EQ(book.best_bid()->price, 10000);
    EXPECT_EQ(book.best_bid()->total_quantity, 50);
    EXPECT_EQ(book.bid_depth(), 1);
}

TEST_F(OrderBookTest, AddAskOrder) {
    auto order = make_limit(Side::Sell, 10100, 30);
    EXPECT_TRUE(book.add(order));
    ASSERT_NE(book.best_ask(), nullptr);
    EXPECT_EQ(book.best_ask()->price, 10100);
    EXPECT_EQ(book.best_ask()->total_quantity, 30);
    EXPECT_EQ(book.ask_depth(), 1);
}

TEST_F(OrderBookTest, PriceTimePriority) {
    // Two orders at same price — first in wins priority
    auto o1 = make_limit(Side::Buy, 10000, 50);
    auto o2 = make_limit(Side::Buy, 10000, 30);
    book.add(o1);
    book.add(o2);

    EXPECT_EQ(book.best_bid()->total_quantity, 80);
    EXPECT_EQ(book.best_bid()->order_count, 2);
}

TEST_F(OrderBookTest, BestBidIsHighestPrice) {
    book.add(make_limit(Side::Buy, 10000, 50));
    book.add(make_limit(Side::Buy, 10200, 30));
    book.add(make_limit(Side::Buy, 10100, 20));

    EXPECT_EQ(book.best_bid()->price, 10200);
}

TEST_F(OrderBookTest, BestAskIsLowestPrice) {
    book.add(make_limit(Side::Sell, 10300, 50));
    book.add(make_limit(Side::Sell, 10100, 30));
    book.add(make_limit(Side::Sell, 10200, 20));

    EXPECT_EQ(book.best_ask()->price, 10100);
}

TEST_F(OrderBookTest, CancelOrder) {
    auto order = make_limit(Side::Buy, 10000, 50);
    book.add(order);
    EXPECT_EQ(book.bid_depth(), 1);

    auto result = book.cancel(order.id);
    EXPECT_EQ(result, CancelResult::Success);
    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_EQ(book.best_bid(), nullptr);
}

TEST_F(OrderBookTest, CancelNonexistent) {
    auto result = book.cancel(99999);
    EXPECT_EQ(result, CancelResult::OrderNotFound);
}

TEST_F(OrderBookTest, Spread) {
    book.add(make_limit(Side::Buy, 10000, 50));
    book.add(make_limit(Side::Sell, 10100, 30));
    EXPECT_EQ(book.spread(), 100);
}

TEST_F(OrderBookTest, L2SnapshotEmptyBook) {
    EXPECT_TRUE(book.l2_bids(5).empty());
    EXPECT_TRUE(book.l2_asks(5).empty());
}

TEST_F(OrderBookTest, L2BidsAreBestFirst) {
    book.add(make_limit(Side::Buy, 9900, 10));
    book.add(make_limit(Side::Buy, 10000, 20));
    book.add(make_limit(Side::Buy, 9800, 30));

    auto levels = book.l2_bids(5);
    ASSERT_EQ(levels.size(), 3u);
    EXPECT_EQ(levels[0].price, 10000);
    EXPECT_EQ(levels[1].price, 9900);
    EXPECT_EQ(levels[2].price, 9800);
}

TEST_F(OrderBookTest, L2AsksAreBestFirst) {
    book.add(make_limit(Side::Sell, 10200, 10));
    book.add(make_limit(Side::Sell, 10100, 20));
    book.add(make_limit(Side::Sell, 10300, 30));

    auto levels = book.l2_asks(5);
    ASSERT_EQ(levels.size(), 3u);
    EXPECT_EQ(levels[0].price, 10100);
    EXPECT_EQ(levels[1].price, 10200);
    EXPECT_EQ(levels[2].price, 10300);
}

TEST_F(OrderBookTest, L2ReportsQuantityNotLevelCount) {
    // The distinction bid_depth() does not make: one level, lots of size.
    book.add(make_limit(Side::Buy, 10000, 400));
    book.add(make_limit(Side::Buy, 10000, 600));

    EXPECT_EQ(book.bid_depth(), 1u);
    auto levels = book.l2_bids(5);
    ASSERT_EQ(levels.size(), 1u);
    EXPECT_EQ(levels[0].total_quantity, 1000);
    EXPECT_EQ(levels[0].order_count, 2u);
}

TEST_F(OrderBookTest, L2TruncatesToMaxLevels) {
    for (Price p = 10000; p > 9000; p -= 100) {
        book.add(make_limit(Side::Buy, p, 10));
    }
    EXPECT_EQ(book.bid_depth(), 10u);
    EXPECT_EQ(book.l2_bids(3).size(), 3u);
    EXPECT_EQ(book.l2_bids(0).size(), 0u);
}

TEST_F(OrderBookTest, L2ReflectsCancels) {
    auto order = make_limit(Side::Buy, 10000, 50);
    book.add(order);
    book.add(make_limit(Side::Buy, 10000, 25));
    EXPECT_EQ(book.l2_bids(1)[0].total_quantity, 75);

    book.cancel(order.id);
    ASSERT_EQ(book.l2_bids(1).size(), 1u);
    EXPECT_EQ(book.l2_bids(1)[0].total_quantity, 25);
    EXPECT_EQ(book.l2_bids(1)[0].order_count, 1u);
}
