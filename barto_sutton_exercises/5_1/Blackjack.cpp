#include "Blackjack.h"

#include <matplot/matplot.h>

#include "barto_sutton_exercises/5_1/Blackjack.h"
#include "m_utils.h"

void Blackjack::initialize() {
    for (int player_sum = MIN_PLAYER_SUM; player_sum < MAX_SUM; player_sum++) {
        for (int dealer_card = ACE; dealer_card <= FACE_CARD; dealer_card++) {
            for (bool usable_ace : {true, false}) {
                State s = {player_sum, dealer_card, usable_ace};
                m_S.push_back(s);

                for (bool hit : {true, false}) {
                    m_A[s].push_back(hit);
                }
            }
        }
    }
}

int Blackjack::draw_card() { return random_value(ACE, FACE_CARD); }

bool Blackjack::is_terminal(const State& s) { return s == dummy_terminal_state; }

std::tuple<int, bool, int> Blackjack::draw_card_with_checks(int player_sum, bool usable_ace) {
    int card = draw_card();
    player_sum += card;

    if (card == ACE && player_sum + USABLE_ACE_VALUE_DIFF <= MAX_SUM) {
        player_sum += USABLE_ACE_VALUE_DIFF;
        usable_ace = true;
    } else if (player_sum > MAX_SUM && usable_ace) {
        player_sum -= USABLE_ACE_VALUE_DIFF;
        usable_ace = false;
    }

    return {player_sum, usable_ace, card};
}

State Blackjack::reset() {
    m_dealer_sum = 0;
    m_dealer_has_usable_ace = false;
    int dealer_card = 0;

    int player_sum = 0;
    bool usable_ace = false;
    int player_card = 0;

    std::tie(player_sum, usable_ace, player_card) = draw_card_with_checks(player_sum, usable_ace);
    std::tie(player_sum, usable_ace, player_card) = draw_card_with_checks(player_sum, usable_ace);

    while (player_sum < MIN_PLAYER_SUM) {
        std::tie(player_sum, usable_ace, player_card) = draw_card_with_checks(player_sum, usable_ace);
    }

    // make sure a natural cannot occur for the player
    while (player_sum == MAX_SUM) {
        player_sum -= player_card;
        std::tie(player_sum, usable_ace, player_card) = draw_card_with_checks(player_sum, usable_ace);
    }

    std::tie(m_dealer_sum, m_dealer_has_usable_ace, dealer_card) =
        draw_card_with_checks(m_dealer_sum, m_dealer_has_usable_ace);

    int dealer_face_up = dealer_card;

    std::tie(m_dealer_sum, m_dealer_has_usable_ace, dealer_card) =
        draw_card_with_checks(m_dealer_sum, m_dealer_has_usable_ace);

    // make sure a natural cannot occur for the dealer
    while (m_dealer_sum == MAX_SUM) {
        m_dealer_sum -= dealer_card;
        std::tie(m_dealer_sum, m_dealer_has_usable_ace, dealer_card) =
            draw_card_with_checks(m_dealer_sum, m_dealer_has_usable_ace);
    }

    return {player_sum, dealer_face_up, usable_ace};
}

std::pair<State, Reward> Blackjack::step(const State& state, const Action& hit) {
    auto [player_sum, face_up_card, usable_ace] = state;

    int player_card = 0;  // not used, just for std::tie to work
    int dealer_card = 0;  // not used, just for std::tie to work

    if (hit) {
        std::tie(player_sum, usable_ace, player_card) = draw_card_with_checks(player_sum, usable_ace);

        if (player_sum > MAX_SUM) {
            return {dummy_terminal_state, LOSS_REWARD};
        } else if (player_sum < MAX_SUM) {
            return {{player_sum, face_up_card, usable_ace}, NO_REWARD};
        }  // else (i.e. player_sum == MAX_SUM), just transition to dealer's turn (fall through)
    }

    // Handle the dealer's turn if the player sticks
    while (m_dealer_sum < DEALER_POLICY_THRESHOLD) {
        std::tie(m_dealer_sum, m_dealer_has_usable_ace, dealer_card) =
            draw_card_with_checks(m_dealer_sum, m_dealer_has_usable_ace);
    }

    // Determine the outcome of the game
    if (m_dealer_sum > MAX_SUM || player_sum > m_dealer_sum) {
        return {dummy_terminal_state, WIN_REWARD};  // Player wins
    } else if (m_dealer_sum == player_sum) {
        return {dummy_terminal_state, NO_REWARD};  // Draw
    } else {
        return {dummy_terminal_state, LOSS_REWARD};  // Dealer wins
    }
}

void Blackjack::plot_policy(DeterministicPolicy<State, Action>& pi) {}